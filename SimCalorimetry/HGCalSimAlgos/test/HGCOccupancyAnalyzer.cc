#include "HGCOccupancyAnalyzer.h"

#include "DetectorDescription/OfflineDBLoader/interface/GeometryInfoDump.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "SimG4CMS/Calo/interface/CaloHitID.h"

#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"

#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "CLHEP/Geometry/Point3D.h"
#include "CLHEP/Geometry/Plane3D.h"
#include "CLHEP/Geometry/Vector3D.h"


#include "FWCore/ParameterSet/interface/FileInPath.h"

#include <fstream>
#include <iostream>

using namespace std;


//
// PLUGIN IMPLEMENTATION
//


//
HGCOccupancyAnalyzer::HGCOccupancyAnalyzer( const edm::ParameterSet &iConfig ) :   
  geoCEE_("HGCalEESensitive"),
  geoCEH_("HGCalHESiliconSensitive"),
  digisCEE_( consumes<HGCalDigiCollection>(edm::InputTag("simHGCalUnsuppressedDigis","EE")) ),
  digisCEH_( consumes<HGCalDigiCollection>(edm::InputTag("simHGCalUnsuppressedDigis","HEfront")) )
{  
  // mipEqThr_=iConfig.getParameter<float>("mipEqThr");
  //fudgeFactor_=iConfig.getParameter<float>("fudgeFactor");
  mipEqThr_=0.5;
  fudgeFactor_=0.8;

  //parse u-v equivalence map file
  edm::FileInPath uvmapF("SimCalorimetry/HGCalSimAlgos/data/uvequiv.dat");
  std::ifstream inF(uvmapF.fullPath());  
  int u,v,sec,ueq,veq;
  while(inF >> u >> v >> sec >> ueq >> veq ) {
    std::pair<int,int> key(u,v),keyeq(ueq,veq);
    uvEqSet_.insert(keyeq);
    uvEqMap_[key] = keyeq;
    uvSectorMap_[key]=sec;
  }

  //ideally these should be read from hgcalDigitzer_cfi.py but ok
  uint32_t adc_nBits(10);
  double adc_saturation(100.);
  adcLSB_=(adc_saturation/pow(2.,adc_nBits));   

  uint32_t tdc_nBits(12);
  double tdc_saturation(10000.);
  tdcLSB_=(tdc_saturation/pow(2.,tdc_nBits)); 
  tdcOnset_=60.;
}

//
HGCOccupancyAnalyzer::~HGCOccupancyAnalyzer()
{
}

//
void HGCOccupancyAnalyzer::endJob()
{
  for(auto &it : waferHistos_) it.second->endJob();
}

//
void HGCOccupancyAnalyzer::prepareAnalysis()
{

  //init histograms
  edm::Service<TFileService> fs;
  ofstream wafer_pos("wafer_pos.dat");
  for(auto &it : hgcGeometries_ )
    {
      const HGCalDDDConstants &ddd=it.second->topology().dddConstants();

      //loop over layers and add wafers in the first sector for each one
      int subdet(it.first=="CEE" ? 0 : 1);
      int nlay( ddd.layers(true) );

      for(int ilay=1; ilay<=nlay; ilay++){

        std::pair<int,int> key(subdet,ilay);
        std::vector<TH1F *> occHistos;
        for(int iwaf=0; iwaf<7; iwaf++) {
          TString name(Form("sd%d_lay%d_hottestwafer%d",subdet,ilay,iwaf));
          occHistos.push_back( fs->make<TH1F>(name,";Occupancy;",500,0,1) );
          occHistos[iwaf]->Sumw2();
        }
        hottestWaferH_[key]=occHistos;
            
        for(auto &uv : uvEqSet_){
          int waferU(uv.first),waferV(uv.second);

          int ncells= ddd.numberCellsHexagon(ilay,waferU,waferV,true);
          if(ncells==0) continue;

          //check geometry
          std::pair<double, double> xy=ddd.waferPosition(waferU,waferV,true);
          double radius( hypot(xy.first,xy.second) );
          if(radius<1) continue;
          double z(ddd.waferZ(ilay,true));
          double eta(TMath::ASinH(z/radius));
          double phi(TMath::ATan2(xy.second,xy.first));
                   
          wafer_pos << subdet << " " << ilay << " "  << waferU << " " << waferV << " " 
                    << ncells << " " << radius << " " << z << " " << eta << " " << phi << endl;

          WaferEquivalentId_t key(std::make_tuple(subdet,ilay,waferU,waferV));
          waferHistos_[key]=new WaferOccupancyHisto(subdet,ilay,waferU,waferV,ncells,&fs);

          //add all the wafer equivalents which this wafer should represent         
          for(auto uveq : uvEqMap_) {
            if(uv!=uveq.second) continue;
            int waferUeq(uveq.first.first), waferVeq(uveq.first.second);
            waferHistos_[key]->addWaferEquivalent(waferUeq,waferVeq);
          }
        }
      }        
    }

  wafer_pos.close();
}

  
//
void HGCOccupancyAnalyzer::analyze( const edm::Event &iEvent, const edm::EventSetup &iSetup)
{
  //read geometry from event setup
  edm::ESHandle<HGCalGeometry> ceeGeoHandle;
  iSetup.get<IdealGeometryRecord>().get(geoCEE_,ceeGeoHandle);
  hgcGeometries_["CEE"]=ceeGeoHandle.product();
  edm::ESHandle<HGCalGeometry> cehGeoHandle;
  iSetup.get<IdealGeometryRecord>().get(geoCEH_,cehGeoHandle);
  hgcGeometries_["CEH"]=cehGeoHandle.product();

  //check if histos need to be instantiated
  if(waferHistos_.size()==0) prepareAnalysis();

  //analyze digi collections
  edm::Handle<HGCalDigiCollection> ceeDigisHandle;
  iEvent.getByToken(digisCEE_,ceeDigisHandle);
  analyzeDigis(0,ceeDigisHandle);
  
  edm::Handle<HGCalDigiCollection> cehDigisHandle;
  iEvent.getByToken(digisCEH_,cehDigisHandle);
  analyzeDigis(1,cehDigisHandle);

  //fill wafer histos and save the max. found in each layer
  std::map<std::pair<int,int>,std::pair<WaferOccupancyHisto::UVKey_t, float> > hotWaferOccPerLayer;
  for(auto &it : waferHistos_) {
    it.second->analyze();

    WaferOccupancyHisto::UVKey_t hotWaferUV=it.second->getHotWaferUV();
    float hotWaferOcc=float(it.second->getHotWaferCounts())/float(it.second->getCells());    
    int sd=std::get<0>(it.first);
    int lay=std::get<1>(it.first);
    std::pair<int,int> key(sd,lay);
   
    if(hotWaferOccPerLayer.find(key)==hotWaferOccPerLayer.end() || hotWaferOccPerLayer[key].second<hotWaferOcc)       
      hotWaferOccPerLayer[key]=std::pair<WaferOccupancyHisto::UVKey_t,float>(hotWaferUV,hotWaferOcc);
  }
    
  //fill the max. counts per layer and in the neighboring cells
  for(auto &it : hotWaferOccPerLayer) {

    int sd=it.first.first;
    int lay=it.first.second;
    WaferOccupancyHisto::UVKey_t hotWaferUV=it.second.first;
    if(hotWaferUV.first==0 && hotWaferUV.second==0) continue; //empty layer

    float hotOcc=it.second.second;
    std::pair<int,int> key(sd,lay);
    hottestWaferH_[key][0]->Fill(hotOcc);

    //find neighbors
    std::vector<float> neighborOccs;
    for(auto &jt : waferHistos_) {
      int isd=std::get<0>(jt.first);
      if(isd!=sd) continue;
      int ilay=std::get<1>(jt.first);
      if(ilay!=lay) continue;
      int neighborCts( jt.second->getNeighborCounts(hotWaferUV) );
      if(neighborCts<0) continue;
      float occ=(float)neighborCts;
      float ncells=(float)jt.second->getCells();
      occ/=ncells;
      neighborOccs.push_back(occ);
    }

    //sort and fill neighbor histos
    std::sort(neighborOccs.begin(),neighborOccs.end(),std::greater<int>());
    for(size_t i=0; i<neighborOccs.size(); i++) {
      hottestWaferH_[key][i+1]->Fill( neighborOccs[i] );
    }
  }
  

 
  //all done, reset counters
  for(auto &it : waferHistos_) it.second->resetCounters();
}


//
void HGCOccupancyAnalyzer::analyzeDigis(int isd,edm::Handle<HGCalDigiCollection> &digiColl)
{
  //check inputs
  if(!digiColl.isValid()) return;

  const HGCalGeometry *geom=hgcGeometries_[isd==0 ? "CEE" : "CEH"];
  
  double qPerMipPerMicron(73*1.602177E-4);
  double mipCorr[3]={1./(qPerMipPerMicron*120.),1./(qPerMipPerMicron*200.),1./(qPerMipPerMicron*300.)};

  //get ddd constants to get cell properties
  const HGCalDDDConstants &ddd=geom->topology().dddConstants();

  //analyze hits
  for(auto &hit : *digiColl)
    {
      if(hit.size()==0) continue;
      int idx=2;

      //detid info
      HGCSiliconDetId detId(hit.id());
      int layer=detId.layer();

      int waferTypeL = ddd.waferType(detId);

      std::pair<int,int> waferUV=detId.waferUV();
      std::pair<int,int> uvEq=uvEqMap_[waferUV];

      //correct ADC by the readoutmode (ADC or TDC)
      double q_mipeq( hit.sample(idx).data() );
      if(hit.sample(idx).mode()){
        q_mipeq = (std::floor(tdcOnset_/adcLSB_)+1.0)*adcLSB_ + (q_mipeq+0.5)*tdcLSB_;
      }else {
        q_mipeq *= adcLSB_;        
      }  

      //normalize to expected MIP response
      q_mipeq = q_mipeq*mipCorr[waferTypeL-1];

      WaferEquivalentId_t key(std::make_tuple(isd,layer,uvEq.first,uvEq.second));
      waferHistos_[key]->count(waferUV.first,waferUV.second,q_mipeq,mipEqThr_,fudgeFactor_);
    }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCOccupancyAnalyzer);
