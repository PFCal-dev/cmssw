#ifndef _HGCOccupancyAnalyzer_h_
#define _HGCOccupancyAnalyzer_h_

#include "interface/WaferOccupancyHisto.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerBase.h"  

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"

#include <string>


/**
   @class HGCOccupancyAnalyzer
   @short an EDAnalyzer to readout the digis in the event and steer the filling of the occupancy histograms
*/

class HGCOccupancyAnalyzer : public edm::EDAnalyzer 
{
  
 public:
  
  explicit HGCOccupancyAnalyzer( const edm::ParameterSet& );
  ~HGCOccupancyAnalyzer();
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
  void endJob();

 private:
  /**
     @short starts histograms
   */
  void prepareAnalysis();
  
  /**
     @short analyze a digi collection
   */
  void analyzeDigis(int ,edm::Handle<HGCalDigiCollection> &);
  
  //containers to hold the equivalence maps of different wafers to the ones in the first sector
  std::set<std::pair<int,int> > uvEqSet_;
  std::map<std::pair<int,int>,std::pair<int,int> > uvEqMap_;
  std::map<std::pair<int,int>,int> uvSectorMap_;

  //histograms for the wafers
  typedef std::tuple<int,int,int,int> WaferEquivalentId_t;
  std::map<WaferEquivalentId_t,WaferOccupancyHisto *> waferHistos_;

  //geometry and digis to analyze
  std::string geoCEE_,geoCEH_;
  std::map<std::string,const HGCalGeometry *> hgcGeometries_;
  edm::EDGetTokenT<HGCalDigiCollection> digisCEE_,digisCEH_;  

  //LSBs and TDC onset
  double adcLSB_,tdcLSB_,tdcOnset_;

  std::map<std::pair<int,int>, std::vector<TH1F *> > hottestWaferH_;

  //thresholds to use for occupancy
  float mipEqThr_,fudgeFactor_;
};
 

#endif
