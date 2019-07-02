#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerBase.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"

#include <cstdint>

#include <cuda.h>
#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerBase.cuh"

using namespace hgc_digi;
using namespace hgc_digi_utils;

template<class DFr>
HGCDigitizerBase<DFr>::HGCDigitizerBase(const edm::ParameterSet& ps) {
  bxTime_        = ps.getParameter<double>("bxTime");
  myCfg_         = ps.getParameter<edm::ParameterSet>("digiCfg");
  doTimeSamples_ = myCfg_.getParameter< bool >("doTimeSamples");
  if(myCfg_.exists("keV2fC"))   keV2fC_   = myCfg_.getParameter<double>("keV2fC");
  else                          keV2fC_   = 1.0;

  if( myCfg_.existsAs<edm::ParameterSet>( "chargeCollectionEfficiencies" ) ) {
    cce_ = myCfg_.getParameter<edm::ParameterSet>("chargeCollectionEfficiencies").template getParameter<std::vector<double>>("values");
  }

  if(myCfg_.existsAs<double>("noise_fC")) {
    noise_fC_.reserve(1);
    noise_fC_.push_back(myCfg_.getParameter<double>("noise_fC"));
  } else if ( myCfg_.existsAs<std::vector<double> >("noise_fC") ) {
    const auto& noises = myCfg_.getParameter<std::vector<double> >("noise_fC");
    noise_fC_ = std::vector<float>(noises.begin(),noises.end());
  } else if(myCfg_.existsAs<edm::ParameterSet>("noise_fC")) {
    const auto& noises = myCfg_.getParameter<edm::ParameterSet>("noise_fC").template getParameter<std::vector<double> >("values");
    noise_fC_ = std::vector<float>(noises.begin(),noises.end());
  } else {
    noise_fC_.resize(1,1.f);
  }
  edm::ParameterSet feCfg = myCfg_.getParameter<edm::ParameterSet>("feCfg");
  myFEelectronics_        = std::unique_ptr<HGCFEElectronics<DFr> >( new HGCFEElectronics<DFr>(feCfg) );
  myFEelectronics_->SetNoiseValues(noise_fC_);

  adcThreshold_fC_  = feCfg.getParameter<double>("adcThreshold_fC");
  adcSaturation_fC_ = feCfg.getParameter<double>("adcSaturation_fC");
  adcNbits_         = feCfg.getParameter<uint32_t>("adcNbits");

}

template<class DFr>
void HGCDigitizerBase<DFr>::run( std::unique_ptr<HGCDigitizerBase::DColl> &digiColl,
				 HGCSimHitDataAccumulator &simData,
				 const CaloSubdetectorGeometry* theGeom,
				 const std::unordered_set<DetId>& validIds,
				 uint32_t digitizationType,
				 CLHEP::HepRandomEngine* engine) {
  if(digitizationType==0) {
    runSimple(digiColl,simData,theGeom,validIds,engine);
    runSimpleOnGPU(digiColl,simData,theGeom,validIds);
  }
  else {
    runDigitizer(digiColl,simData,theGeom,validIds,digitizationType,engine);
  }
}

template<class DFr>
void HGCDigitizerBase<DFr>::runSimple(std::unique_ptr<HGCDigitizerBase::DColl> &coll,
				      HGCSimHitDataAccumulator &simData,
				      const CaloSubdetectorGeometry* theGeom,
				      const std::unordered_set<DetId>& validIds,
				      CLHEP::HepRandomEngine* engine) {
  HGCSimHitData chargeColl,toa;

  // this represents a cell with no signal charge
  HGCCellInfo zeroData;
  zeroData.hit_info[0].fill(0.f); //accumulated energy
  zeroData.hit_info[1].fill(0.f); //time-of-flight

  for( const auto& id : validIds ) {
    chargeColl.fill(0.f);
    toa.fill(0.f);
    HGCSimHitDataAccumulator::iterator it = simData.find(id);
    HGCCellInfo& cell = ( simData.end() == it ? zeroData : it->second );
    addCellMetadata(cell,theGeom,id);

    for(size_t i=0; i<cell.hit_info[0].size(); i++) {
      double rawCharge(cell.hit_info[0][i]);

      //time of arrival
      toa[i]=cell.hit_info[1][i];
      if(myFEelectronics_->toaMode()==HGCFEElectronics<DFr>::WEIGHTEDBYE && rawCharge>0)
        toa[i]=cell.hit_info[1][i]/rawCharge;

      //convert total energy in GeV to charge (fC)
      //double totalEn=rawEn*1e6*keV2fC_;
      float totalCharge=rawCharge;

      //add noise (in fC)
      //we assume it's randomly distributed and won't impact ToA measurement
      //also assume that it is related to the charge path only and that noise fluctuation for ToA circuit be handled separately
      if (noise_fC_[cell.thickness-1] != 0)
        totalCharge += std::max( (float)CLHEP::RandGaussQ::shoot(engine,0.0,cell.size*noise_fC_[cell.thickness-1]) , 0.f );
      if(totalCharge<0.f) totalCharge=0.f;

      chargeColl[i]= totalCharge;
    }

    //run the shaper to create a new data frame
    DFr rawDataFrame( id );
    if( !cce_.empty() )
      myFEelectronics_->runShaper(rawDataFrame, chargeColl, toa, cell.thickness, engine, cce_[cell.thickness-1]);
    else
      myFEelectronics_->runShaper(rawDataFrame, chargeColl, toa, cell.thickness, engine);

    //update the output according to the final shape
    updateOutput(coll,rawDataFrame);
  }
}

template<class DFr>
void HGCDigitizerBase<DFr>::runSimpleOnGPU(std::unique_ptr<HGCDigitizerBase::DColl> &coll,
				      HGCSimHitDataAccumulator &simData,
				      const CaloSubdetectorGeometry* theGeom,
				      const std::unordered_set<DetId>& validIds) {

  const size_t Nbx(5); //this is hardcoded
  const uint32_t N(Nbx*validIds.size());

  //host arrays
  float *toa     = (float*)malloc(N*sizeof(float));
  float *charge  = (float*)malloc(N*sizeof(float));
  uint8_t *type  = (uint8_t*)malloc(N*sizeof(uint8_t));
  uint32_t *rawData = (uint32_t*)malloc(N*sizeof(uint32_t));
  uint32_t idIdx(0);
  for( const auto& id : validIds ) {

    HGCSimHitDataAccumulator::iterator it = simData.find(id);
    const HGCCellInfo *cell( simData.end() == it ? NULL : &(it->second) );

    for(size_t i=0; i<Nbx; i++) {
      uint32_t arrIdx(idIdx*Nbx+i);
      if(cell!=NULL){
        charge[arrIdx] = cell->hit_info[0][i];
        toa[arrIdx]    = cell->hit_info[1][i];
        type[arrIdx]   = (uint8_t)(cell->thickness - 100)/100;
      }
      else{
        charge[arrIdx]=0.f;
        toa[arrIdx]=0.f;
      }
      idIdx++;
    }
  }

  //device arrays
  float *d_toa, *d_charge;
  uint8_t* d_type;
  uint32_t *d_rawData;
  cudaMalloc(&d_toa,     N*sizeof(float));
  cudaMalloc(&d_charge,  N*sizeof(float));
  cudaMalloc(&d_type,    N*sizeof(uint8_t));
  cudaMalloc(&d_rawData, N*sizeof(uint32_t));
  cudaMemcpy(d_toa,    toa,    N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_charge, charge, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_type,   type,   N*sizeof(uint8_t), cudaMemcpyHostToDevice);

  //allocate N rand floats on the GPU
  float *devRand;
  cudaMalloc((void **)&devRand, N * sizeof(float));


  //call function on the GPU
  addNoiseWrapper(N, d_charge, d_toa, toaModeByEnergy(), devRand, d_type, d_rawData);


  //copy back result and add to the event
  cudaMemcpy(rawData, d_rawData, N*sizeof(uint32_t), cudaMemcpyDeviceToHost);
  updateOutput(validIds, rawData,coll);

  //free memory
  cudaFree(d_toa);
  cudaFree(d_charge);
  cudaFree(d_rawData);
  free(toa);
  free(charge);
  free(rawData);
}




template<class DFr>
void HGCDigitizerBase<DFr>::updateOutput(std::unique_ptr<HGCDigitizerBase::DColl> &coll,
                                          const DFr& rawDataFrame) {
  int itIdx(9);
  if(rawDataFrame.size()<=itIdx+2) return;

  DFr dataFrame( rawDataFrame.id() );
  dataFrame.resize(5);
  bool putInEvent(false);
  for(int it=0;it<5; it++) {
    dataFrame.setSample(it, rawDataFrame[itIdx-2+it]);
    if(it==2) putInEvent = rawDataFrame[itIdx-2+it].threshold();
  }

  if(putInEvent) {
    coll->push_back(dataFrame);
  }
}


template<class DFr>
void HGCDigitizerBase<DFr>::updateOutput(const std::unordered_set<DetId>& validIds,
                                         const uint32_t *bxWord,
                                         std::unique_ptr<HGCDigitizerBase::DColl> &coll){
  unsigned long idx(0);
  for( const auto& id : validIds ) {

    bool putInEvent(false);
    DFr dataFrame( id );
    dataFrame.resize(5);

    //loop over 5 bunches and fill the dataframe
    uint32_t idxT=idx*5;
    for(size_t bx=0; bx<5; bx++) {
      idxT+=bx;
      dataFrame.setSample(bx, bxWord[idxT]);
      if(bx!=2) continue;
      putInEvent=( (bxWord[idxT] >> HGCSample::kThreshShift) & HGCSample::kThreshMask );
    }

    //add to event if in-time bunch charge has passed the threshold
    if(putInEvent)
      coll->push_back(dataFrame);

    ++idx;
  }
}

// cause the compiler to generate the appropriate code
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
template class HGCDigitizerBase<HGCEEDataFrame>;
template class HGCDigitizerBase<HGCBHDataFrame>;
template class HGCDigitizerBase<HGCalDataFrame>;
