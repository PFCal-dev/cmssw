#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerNtupleBase.h"



class HGCalTriggerNtupleHGCTowerMaps : public HGCalTriggerNtupleBase
{

  public:
    HGCalTriggerNtupleHGCTowerMaps(const edm::ParameterSet& conf);
    ~HGCalTriggerNtupleHGCTowerMaps(){};
    virtual void initialize(TTree&, const edm::ParameterSet&, edm::ConsumesCollector&&) override final;
    virtual void fill(const edm::Event& e, const edm::EventSetup& es) override final;

  private:
    virtual void clear() override final;

    edm::EDGetToken towermaps_token_;

    int towermap_n_ ;
    std::vector<uint32_t> towermap_id_;
    std::vector<float> towermap_pt_;
    std::vector<float> towermap_energy_;
    std::vector<float> towermap_eta_;
    std::vector<float> towermap_phi_;
    std::vector<int> towermap_const_n_;
};

DEFINE_EDM_PLUGIN(HGCalTriggerNtupleFactory,
    HGCalTriggerNtupleHGCTowerMaps,
    "HGCalTriggerNtupleHGCTowerMaps" );


HGCalTriggerNtupleHGCTowerMaps::
HGCalTriggerNtupleHGCTowerMaps(const edm::ParameterSet& conf):HGCalTriggerNtupleBase(conf)
{
}

void
HGCalTriggerNtupleHGCTowerMaps::
initialize(TTree& tree, const edm::ParameterSet& conf, edm::ConsumesCollector&& collector)
{
  towermaps_token_ = collector.consumes<l1t::HGCalMulticlusterBxCollection>(conf.getParameter<edm::InputTag>("TowerMaps"));

  tree.Branch("towermap_n", &towermap_n_, "towermap_n/I");
  tree.Branch("towermap_id", &towermap_id_);
  tree.Branch("towermap_pt", &towermap_pt_);
  tree.Branch("towermap_energy", &towermap_energy_);
  tree.Branch("towermap_eta", &towermap_eta_);
  tree.Branch("towermap_phi", &towermap_phi_);
  tree.Branch("towermap_const_n", &towermap_const_n_);

}

void
HGCalTriggerNtupleHGCTowerMaps::
fill(const edm::Event& e, const edm::EventSetup& es)
{

  // retrieve towermaps
  edm::Handle<l1t::HGCalMulticlusterBxCollection> towermaps_h;
  e.getByToken(towermaps_token_, towermaps_h);
  const l1t::HGCalMulticlusterBxCollection& towermaps = *towermaps_h;

  // retrieve geometry
  edm::ESHandle<HGCalTriggerGeometryBase> geometry;
  es.get<CaloGeometryRecord>().get(geometry);

  clear();
  for(auto towermap_itr=towermaps.begin(0); towermap_itr!=towermaps.end(0); towermap_itr++)
  {
    towermap_n_++;
    towermap_id_.emplace_back(towermap_itr->detId());
    // physical values 
    towermap_pt_.emplace_back(towermap_itr->pt());
    towermap_energy_.emplace_back(towermap_itr->energy());
    towermap_eta_.emplace_back(towermap_itr->eta());
    towermap_phi_.emplace_back(towermap_itr->phi());
    towermap_const_n_.emplace_back(towermap_itr->constituents().size());    
  }
}


void
HGCalTriggerNtupleHGCTowerMaps::
clear()
{
  towermap_n_ = 0;
  towermap_id_.clear();
  towermap_pt_.clear();
  towermap_energy_.clear();
  towermap_eta_.clear();
  towermap_phi_.clear();
  towermap_const_n_.clear();
}




