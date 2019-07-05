#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerBase.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"

#include <cstdint>
#include <chrono>

#include <cuda.h>
#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerBase.cuh"

using namespace hgc_digi;
using namespace hgc_digi_utils;

template<class DFr>
HGCDigitizerBase<DFr>::HGCDigitizerBase(const edm::ParameterSet& ps) : isCUDAInit(false) {
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


  std::cout << "--> N channels = " << validIds.size() << std::endl;


  if(digitizationType==0) {
    runSimple(digiColl,simData,theGeom,validIds,engine);
  }
  else if(digitizationType==1) {
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


  auto startCPU = std::chrono::system_clock::now();
  //std::cout << "==>>> runSimple <<===" << std::endl;


  // this represents a cell with no signal charge
  HGCCellInfo zeroData;
  zeroData.hit_info[0].fill(0.f); //accumulated energy
  zeroData.hit_info[1].fill(0.f); //time-of-flight

  const size_t Nbx(5); //this is hardcoded
  const uint32_t N(Nbx*validIds.size());
  h_rawData = (uint32_t*)malloc(N*sizeof(uint32_t));

  int globalCount = 0;
  for( const auto& id : validIds ) {
    HGCSimHitDataAccumulator::iterator it = simData.find(id);
    HGCCellInfo& cell = ( simData.end() == it ? zeroData : it->second );
    addCellMetadata(cell,theGeom,id);

    //std::cout << "==>> " << cell.thickness << std::endl;

    float noise[] = {0.168,0.336,0.256};


    for(size_t i=0; i<Nbx; i++) {
      double rawCharge(cell.hit_info[0][i+7]);

      //time of arrival
      float toa = cell.hit_info[1][i+7];
      if(toaModeByEnergy() && rawCharge>0)
        toa = toa/rawCharge;

      //convert total energy in GeV to charge (fC)
      //double totalEn=rawEn*1e6*keV2fC_;
      float totalCharge=rawCharge;

      //add noise (in fC)
      //we assume it's randomly distributed and won't impact ToA measurement
      //also assume that it is related to the charge path only and that noise fluctuation for ToA circuit be handled separately
      totalCharge += (float)CLHEP::RandGaussQ::shoot(engine, 0.0, 1.) * noise[cell.thickness-1];
      if(totalCharge<0.f) totalCharge=0.f;

      bool passThr=(totalCharge>0.672);
      uint16_t finalCharge=(uint16_t)(fminf( totalCharge, 100.)/0.0977);
      uint16_t finalToA=(uint16_t)(toa/0.0244);

      //std::cout << "==>> rawCharge, totalCharge, toa " << rawCharge << " " << finalCharge << " " << finalToA << std::endl;

      uint32_t word = ( (passThr<<31) |
                      ((finalToA & 0x3ff) <<13) |
                      ((finalCharge & 0xfff)));

      h_rawData[globalCount] = word;
      ++globalCount;
    }
  }

  auto endCPU = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds_CPU = endCPU-startCPU;
  std::cout << "==>>> CPU time (s): " << elapsed_seconds_CPU.count() << std::endl;

  //update the output according to the final shape
  updateOutput(validIds, h_rawData, coll);
}

template<class DFr>
void HGCDigitizerBase<DFr>::runSimpleOnGPU(std::unique_ptr<HGCDigitizerBase::DColl> &coll,
				      HGCSimHitDataAccumulator &simData,
				      const CaloSubdetectorGeometry* theGeom,
				      const std::unordered_set<DetId>& validIds) {

  auto startGPU = std::chrono::system_clock::now();

  //std::cout << "==>>> runSimpleOnGPU <<===" << std::endl;

  bool debug(false);
  HGCCellInfo zeroData;
  zeroData.hit_info[0].fill(0.f); //accumulated energy
  zeroData.hit_info[1].fill(0.f); //time-of-flight

  const size_t Nbx(5); //this is hardcoded
  const uint32_t N(Nbx*validIds.size());

  initCUDA(N);

  uint32_t idIdx(0);
  if(debug)
  {
    std::cout << "--> size charge: " << (sizeof(h_charge)/sizeof(float)) << std::endl;
    std::cout << "--> size type: " << (sizeof(h_type)/sizeof(uint16_t)) << std::endl;
    std::cout << "--> size rawData: " << (sizeof(h_rawData)/sizeof(uint32_t)) << std::endl;
    std::cout << "==>> validIds.size = " << validIds.size() << std::endl;
  }

  for( const auto& id : validIds ) {

    HGCSimHitDataAccumulator::iterator it = simData.find(id);
    HGCCellInfo* cell( simData.end() == it ? &zeroData : &(it->second) );
    addCellMetadata(*cell, &(*theGeom), id);

    if(debug)
    {
      std::cout << "==> idIdx = " << idIdx << std::endl;
      std::cout << "==>> cell->hit_info[0].size() = " << cell->hit_info[0].size() << std::endl;
      std::cout << "==>> cell->hit_info[1].size() = " << cell->hit_info[1].size() << std::endl;
    }

    for(size_t i=0; i<Nbx; i++) {
      uint32_t arrIdx(idIdx*Nbx+i);
      if(debug)
        std::cout << "==>> arrIdx = " << arrIdx << std::endl;

      if(cell!=NULL){
        h_charge[arrIdx] = cell->hit_info[0][i+7];
        h_toa[arrIdx]    = cell->hit_info[1][i+7];
        h_type[arrIdx]   = cell->thickness-1;

        if(debug)
        {
          std::cout << "==>> cell->charge " << h_charge[arrIdx] << std::endl;
          std::cout << "==>> cell->toa " << h_toa[arrIdx] << std::endl;
          std::cout << "==>> cell->thickness " << h_type[arrIdx] << std::endl;
        }
      }
      else{
        h_charge[arrIdx]=0.f;
        h_toa[arrIdx]=0.f;
      }
    }
    idIdx++;
  }

  //copy to device
  cudaMemcpy(d_toa,    h_toa,    N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_charge, h_charge, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_type,   h_type,   N*sizeof(uint16_t), cudaMemcpyHostToDevice);

  //call function on the GPU
  addNoiseWrapper(N, d_charge, d_toa, toaModeByEnergy(), d_rand, d_type, d_rawData, d_gen);

  //copy back result and add to the event
  cudaMemcpy(h_rawData, d_rawData, N*sizeof(uint32_t), cudaMemcpyDeviceToHost);


  auto endGPU = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds_GPU = endGPU-startGPU;
  std::cout << "==>>> GPU time (s) " << elapsed_seconds_GPU.count() << std::endl;

  updateOutput(validIds, h_rawData, coll);
}

//
template<class DFr>
void HGCDigitizerBase<DFr>::initCUDA(const uint32_t N){
  if(isCUDAInit) return;
  h_toa     = (float*)malloc(N*sizeof(float)); //host arrays
  h_charge  = (float*)malloc(N*sizeof(float));
  h_type    = (uint16_t*)malloc(N*sizeof(uint16_t));
  h_rawData = (uint32_t*)malloc(N*sizeof(uint32_t));
  cudaMalloc(&d_toa,     N*sizeof(float));  //device arrays
  cudaMalloc(&d_charge,  N*sizeof(float));
  cudaMalloc(&d_type,    N*sizeof(uint16_t));
  cudaMalloc(&d_rawData, N*sizeof(uint32_t));
  cudaMalloc(&d_rand,    N*sizeof(float));

  //Create pseudo-random number generator on the device
  curandCreateGenerator(&d_gen, CURAND_RNG_PSEUDO_DEFAULT);

  isCUDAInit=true;
}


//
template<class DFr>
void HGCDigitizerBase<DFr>::endCUDA(){
  //free memory
  cudaFree(d_toa);
  cudaFree(d_charge);
  cudaFree(d_rawData);
  cudaFree(d_type);
  cudaFree(d_rand);
  free(h_toa);
  free(h_charge);
  free(h_rawData);
  free(h_type);
  curandDestroyGenerator(d_gen);
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
  //coll->reserve(50000);
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
