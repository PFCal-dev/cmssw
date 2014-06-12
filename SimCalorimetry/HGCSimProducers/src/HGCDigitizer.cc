#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "SimCalorimetry/HGCSimProducers/interface/HGCDigitizer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Event.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"
#include <boost/foreach.hpp>
#include "Geometry/Records/interface/IdealGeometryRecord.h"

//
HGCDigitizer::HGCDigitizer(const edm::ParameterSet& ps) :
  theHGCEEDigitizer_(ps),
  theHGCHEbackDigitizer_(ps),
  theHGCHEfrontDigitizer_(ps),
  mySubDet_(ForwardSubdetector::ForwardEmpty)
{
  //configure from cfg
  hitCollection_     = ps.getUntrackedParameter< std::string >("hitCollection");
  digiCollection_    = ps.getUntrackedParameter< std::string >("digiCollection");
  maxSimHitsAccTime_ = ps.getUntrackedParameter< uint32_t >("maxSimHitsAccTime");
  bxTime_            = ps.getUntrackedParameter< int32_t >("bxTime");
  doTrivialDigis_    = ps.getUntrackedParameter< bool >("doTrivialDigis");

  //get the random number generator
  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable()) {
    throw cms::Exception("Configuration") << "HGCDigitizer requires the RandomNumberGeneratorService - please add this service or remove the modules that require it";
  }
  CLHEP::HepRandomEngine& engine = rng->getEngine();
  theHGCEEDigitizer_.setRandomNumberEngine(engine);
  theHGCHEbackDigitizer_.setRandomNumberEngine(engine);
  theHGCHEfrontDigitizer_.setRandomNumberEngine(engine);

  //subdetector
  if( producesEEDigis() )      mySubDet_=ForwardSubdetector::HGCEE;
  if( producesHEfrontDigis() ) mySubDet_=ForwardSubdetector::HGCHEF;
  if( producesHEbackDigis() )  mySubDet_=ForwardSubdetector::HGCHEB;
}

//
void HGCDigitizer::initializeEvent(edm::Event const& e, edm::EventSetup const& es)
{
  resetSimHitDataAccumulator(); 
}

//
void HGCDigitizer::finalizeEvent(edm::Event& e, edm::EventSetup const& es)
{
  if( producesEEDigis() ) 
    {
      std::auto_ptr<HGCEEDigiCollection> digiResult(new HGCEEDigiCollection() );
      theHGCEEDigitizer_.run(digiResult,simHitAccumulator_,doTrivialDigis_);
      edm::LogInfo("HGCDigitizer") << " @ finalize event - produced " << digiResult->size() <<  " EE hits";
      e.put(digiResult,digiCollection());
    }
  if( producesHEfrontDigis())
    {
      std::auto_ptr<HGCHEDigiCollection> digiResult(new HGCHEDigiCollection() );
      theHGCHEfrontDigitizer_.run(digiResult,simHitAccumulator_,doTrivialDigis_);
      edm::LogInfo("HGCDigitizer") << " @ finalize event - produced " << digiResult->size() <<  " HE front hits";
      e.put(digiResult,digiCollection());
    }
  if( producesHEbackDigis() )
    {
      std::auto_ptr<HGCHEDigiCollection> digiResult(new HGCHEDigiCollection() );
      theHGCHEbackDigitizer_.run(digiResult,simHitAccumulator_,doTrivialDigis_);
      edm::LogInfo("HGCDigitizer") << " @ finalize event - produced " << digiResult->size() <<  " HE back hits";
      e.put(digiResult,digiCollection());
    }
}

//
void HGCDigitizer::accumulate(edm::Event const& e, edm::EventSetup const& eventSetup) 
{
  //get geometry
  edm::ESHandle<HGCalGeometry> geom;
  std::cout << producesEEDigis() << " " << producesHEfrontDigis() << " " <<  producesHEbackDigis()  << std::endl; 
  if( producesEEDigis() )      eventSetup.get<IdealGeometryRecord>().get("HGCalEESensitive"            , geom);
  if( producesHEfrontDigis() ) eventSetup.get<IdealGeometryRecord>().get("HGCalHESiliconSensitive"     , geom);
  if( producesHEbackDigis() )  eventSetup.get<IdealGeometryRecord>().get("HGCalHEScintillatorSensitive", geom);
  std::cout << geom.isValid() << " " << std::endl;

  //get inputs
  edm::Handle<edm::PCaloHitContainer> hits;
  e.getByLabel(edm::InputTag("g4SimHits",hitCollection_),hits); 
  if( !hits.isValid() ){
    edm::LogError("HGCDigitizer") << " @ accumulate : can't find " << hitCollection_ << " collection of g4SimHits";
    return;
  }

  //accumulate in-time the main event
  accumulate(hits, 0, geom);
}

//
void HGCDigitizer::accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& eventSetup) 
{
  //get geometry
  edm::ESHandle<HGCalGeometry> geom;
  if( producesEEDigis() )      eventSetup.get<IdealGeometryRecord>().get("HGCalEESensitive"            , geom);
  if( producesHEfrontDigis() ) eventSetup.get<IdealGeometryRecord>().get("HGCalHESiliconSensitive"     , geom);
  if( producesHEbackDigis() )  eventSetup.get<IdealGeometryRecord>().get("HGCalHEScintillatorSensitive", geom);

  //get inputs
  edm::Handle<edm::PCaloHitContainer> hits;
  e.getByLabel(edm::InputTag("g4SimHits",hitCollection_),hits); 
  if( !hits.isValid() ){
    edm::LogError("HGCDigitizer") << " @ accumulate : can't find " << hitCollection_ << " collection of g4SimHits";
    return;
  }

  //accumulate for the simulated bunch crossing
  accumulate(hits, e.bunchCrossing(),geom);
}

//
void HGCDigitizer::accumulate(edm::Handle<edm::PCaloHitContainer> const &hits, int bxCrossing,const edm::ESHandle<HGCalGeometry> &geom)
{
  for(edm::PCaloHitContainer::const_iterator hit_it = hits->begin(); hit_it != hits->end(); ++hit_it)
    {
      HGCalDetId simId( hit_it->id() );
      int layer(simId.layer()), cell(simId.cell());
      if(geom.isValid())
	{
	  const HGCalTopology &topo=geom->topology();
	  const HGCalDDDConstants &dddConst=topo.dddConstants();
	  std::pair<int,int> recoLayerCell=dddConst.simToReco(cell,layer,topo.detectorType());
	  std::cout << layer << "," << cell << "->";
	  cell  = recoLayerCell.first;
	  layer = recoLayerCell.second;
	  std::cout << layer << "," << cell << std::endl;
	}

      //this could be changed in the future to use the geometry record
      DetId id = ( producesEEDigis() ?
		   (uint32_t)HGCEEDetId(mySubDet_,simId.zside(),layer,simId.sector(),simId.subsector(),cell):
		   (uint32_t)HGCHEDetId(mySubDet_,simId.zside(),layer,simId.sector(),simId.subsector(),cell) );

      GlobalPoint globalPos=geom->getPosition( id );

      //single time sample
      int    itime = 0; //
      //check units
      //int itime=(int) ( ((hit_it->time()-globalPos.z()/CLHEP::c_light) - bxTime_*bxCrossing)/bxCrossing ); // - jitter etc.;
      double ien   = hit_it->energy();

      HGCSimHitDataAccumulator::iterator simHitIt=simHitAccumulator_.find(id);
      if(simHitIt==simHitAccumulator_.end())
	{
	  HGCSimHitData baseData(10,0);
	  simHitAccumulator_[id]=baseData;
	  simHitIt=simHitAccumulator_.find(id);
	}
      if(itime<0 || itime>(int)simHitIt->second.size()) continue;
      (simHitIt->second)[itime] += ien;
    }
}

//
void HGCDigitizer::beginRun(const edm::EventSetup & es)
{
  //checkGeometry(es);
  //theShapes->beginRun(es);
}

//
void HGCDigitizer::endRun()
{
  //theShapes->endRun();   
}

//
void HGCDigitizer::resetSimHitDataAccumulator()
{
  for( HGCSimHitDataAccumulator::iterator it = simHitAccumulator_.begin(); it!=simHitAccumulator_.end(); it++) 
    std::fill(it->second.begin(), it->second.end(),0.); 
}


//
HGCDigitizer::~HGCDigitizer()
{
}


