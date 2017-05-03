#ifndef DataFormats_L1Trigger_HGCalMulticluster_h
#define DataFormats_L1Trigger_HGCalMulticluster_h

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/L1THGCal/interface/HGCalClusterT.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"

namespace l1t {
            
  class HGCalMulticluster : public HGCalClusterT<l1t::HGCalCluster> {
    
    public:
       
      HGCalMulticluster(){}
      HGCalMulticluster( const LorentzVector p4,
          int pt=0,
          int eta=0,
          int phi=0
          );

      HGCalMulticluster( const edm::Ptr<l1t::HGCalCluster> &tc );
      
      ~HGCalMulticluster();


	int32_t MultiClusterET() const {return  MultiClusterET_;}
	int32_t MultiClusterZ() const {return  MultiClusterZ_;}
	int32_t MultiClusterPhi() const {return MultiClusterPhi_;}
	int32_t MultiClusterEta() const {return MultiClusterEta_;}
	int32_t MultiClusterFlags() const {return MultiClusterFlags_;}
	int32_t MultiClusterQFlags() const {return MultiClusterQFlags_;}
	int32_t MultiClusterNoTriggerCells() const {return MultiClusterNoTriggerCells_;}
	int32_t MultiClusterTBA() const {return MultiClusterTBA_;}
	int32_t MultiClusterMaxEnergyLayer() const {return MultiClusterMaxEnergyLayer_;}

	void SetMultiClusterET(const int32_t MCET ) {  MultiClusterET_ = MCET;}
	void SetMultiClusterZ(const int32_t MCZ) {  MultiClusterZ_ = MCZ;}
	void SetMultiClusterPhi(const int32_t MCP) { MultiClusterPhi_ = MCP;}
	void SetMultiClusterEta(const int32_t MCE ) { MultiClusterEta_ = MCE;}
	void SetMultiClusterFlags(const int32_t MCF ) { MultiClusterFlags_ = MCF;}
	void SetMultiClusterQFlags(const int32_t MCQF) { MultiClusterQFlags_ = MCQF ;}
	void SetMultiClusterNoTriggerCells(const int32_t MCNTC) { MultiClusterNoTriggerCells_ = MCNTC;}
	void SetMultiClusterTBA(const int32_t MCTBA) { MultiClusterTBA_ = MCTBA;}
	void SetMultiClusterMaxEnergyLayer(const int32_t MCMEL) { MultiClusterMaxEnergyLayer_ = MCMEL;}


      
    private:
      
      int32_t MultiClusterET_            = 0;
      int32_t MultiClusterZ_             = 0;
      int32_t MultiClusterPhi_           = 0;
      int32_t MultiClusterEta_           = 0;
      int32_t MultiClusterFlags_         = 0;
      int32_t MultiClusterQFlags_        = 0;
      int32_t MultiClusterNoTriggerCells_= 0;
      int32_t MultiClusterTBA_           = 0;
      int32_t MultiClusterMaxEnergyLayer_= 0;
      
  };
    
  typedef BXVector<HGCalMulticluster> HGCalMulticlusterBxCollection;  
  
}

#endif
