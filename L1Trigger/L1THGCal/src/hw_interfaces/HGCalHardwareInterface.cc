#include "L1Trigger/L1THGCal/interface/hw_interfaces/HGCalHardwareInterface.h"

bool HGCalHardwareInterface::decodeClusterData (const ClusterDataContainer& DataToBeDecoded, l1t::HGCalCluster & HGCC) const {
  if (HGCC.clusterType() == l1t::HGCalClusterT<l1t::HGCalTriggerCell>::ClusterType::Type2D){
    decodeCluster2DData(DataToBeDecoded, HGCC);
  }
  return true;
}

bool HGCalHardwareInterface::decodeClusterData (const ClusterDataContainer& DataToBeDecoded, l1t::HGCalMulticluster & HGCMC) const {
  switch (HGCMC.clusterType()){
  case l1t::HGCalClusterT<l1t::HGCalCluster>::ClusterType::Type3D:
    decodeCluster3DData(DataToBeDecoded, HGCMC);
    break;
  default: 
    return false;
  }
  return true;
}

bool HGCalHardwareInterface::encodeClusterData (ClusterDataContainer& DataToBeEncoded, const l1t::HGCalCluster & HGCC) const {
  switch (HGCC.clusterType()){
  case l1t::HGCalClusterT<l1t::HGCalTriggerCell>::ClusterType::Type2D:
    encodeCluster2DData(DataToBeEncoded, HGCC);
    break;
  default: 
    return false;
  }
  return true;
}

bool HGCalHardwareInterface::encodeClusterData (ClusterDataContainer& DataToBeEncoded, const l1t::HGCalMulticluster & HGCMC) const {
  switch (HGCMC.clusterType()){
  case l1t::HGCalClusterT<l1t::HGCalCluster>::ClusterType::Type3D:
    encodeCluster3DData(DataToBeEncoded, HGCMC);
    break;
  default: 
    return false;
  }
  return true;
}

bool HGCalHardwareInterface::decodeCluster2DData (const ClusterDataContainer& DataToBeDecoded, l1t::HGCalCluster & HGCC) const{
  if(DataToBeDecoded.empty()) {
    return false;
  }
  HGCC.setCluster2DET(            get32bitPart(DataToBeDecoded.at(Cluster2DDataFormat::ET/wordsize_),             
					       Cluster2DDataFormat::ET % wordsize_,             
					       cluster2DDataMap_.at(Cluster2DDataFormat::ET)));
  HGCC.setCluster2DCenterX(       get32bitPart(DataToBeDecoded.at(Cluster2DDataFormat::CenterX/wordsize_),        
					       Cluster2DDataFormat::CenterX % wordsize_,        
					       cluster2DDataMap_.at(Cluster2DDataFormat::CenterX)));
  HGCC.setCluster2DCenterY(       get32bitPart(DataToBeDecoded.at(Cluster2DDataFormat::CenterY/wordsize_),        
					       Cluster2DDataFormat::CenterY % wordsize_,        
					       cluster2DDataMap_.at(Cluster2DDataFormat::CenterY)));
  HGCC.setCluster2DSizeX(         get32bitPart(DataToBeDecoded.at(Cluster2DDataFormat::SizeX/wordsize_),          
					       Cluster2DDataFormat::SizeX % wordsize_,          
					       cluster2DDataMap_.at(Cluster2DDataFormat::SizeX)));
  HGCC.setCluster2DSizeY(         get32bitPart(DataToBeDecoded.at(Cluster2DDataFormat::SizeY/wordsize_),          
					       Cluster2DDataFormat::SizeY % wordsize_,          
					       cluster2DDataMap_.at(Cluster2DDataFormat::SizeY)));
  HGCC.setCluster2DNoTriggerCells(get32bitPart(DataToBeDecoded.at(Cluster2DDataFormat::NoTriggerCells/wordsize_), 
					       Cluster2DDataFormat::NoTriggerCells % wordsize_, 
					       cluster2DDataMap_.at(Cluster2DDataFormat::NoTriggerCells)));
  HGCC.setCluster2DnLocalMaxima(  get32bitPart(DataToBeDecoded.at(Cluster2DDataFormat::LocalMax/wordsize_),       
					       Cluster2DDataFormat::LocalMax % wordsize_,       
					       cluster2DDataMap_.at(Cluster2DDataFormat::LocalMax)));
  /* The break statements are omitted since the cases are inclusive in order */ 
  switch ( HGCC.cluster2DnLocalMaxima()){
  case 3: 
    HGCC.setCluster2DLocalMax3ET(      get32bitPart(DataToBeDecoded.at(Cluster2DDataFormat::LocalMax3ET/wordsize_),   
						    Cluster2DDataFormat::LocalMax3ET % wordsize_,   
						    cluster2DDataMap_.at(Cluster2DDataFormat::LocalMax3ET)));
    HGCC.setCluster2DLocalMax3RelY(    get32bitPart(DataToBeDecoded.at(Cluster2DDataFormat::LocalMax3RelY/wordsize_), 
						    Cluster2DDataFormat::LocalMax3RelY % wordsize_, 
						    cluster2DDataMap_.at(Cluster2DDataFormat::LocalMax3RelY)));
    HGCC.setCluster2DLocalMax3RelX(    get32bitPart(DataToBeDecoded.at(Cluster2DDataFormat::LocalMax3RelX/wordsize_), 
						    Cluster2DDataFormat::LocalMax3RelX % wordsize_, 
						    cluster2DDataMap_.at(Cluster2DDataFormat::LocalMax3RelX)));
  case 2:
    HGCC.setCluster2DLocalMax2ET(      get32bitPart(DataToBeDecoded.at(Cluster2DDataFormat::LocalMax2ET/wordsize_),   
						    Cluster2DDataFormat::LocalMax2ET % wordsize_,   
						    cluster2DDataMap_.at(Cluster2DDataFormat::LocalMax2ET)));
    HGCC.setCluster2DLocalMax2RelY(    get32bitPart(DataToBeDecoded.at(Cluster2DDataFormat::LocalMax2RelY/wordsize_), 
						    Cluster2DDataFormat::LocalMax2RelY % wordsize_, 
						    cluster2DDataMap_.at(Cluster2DDataFormat::LocalMax2RelY)));
    HGCC.setCluster2DLocalMax2RelX(    get32bitPart(DataToBeDecoded.at(Cluster2DDataFormat::LocalMax2RelX/wordsize_), 
						    Cluster2DDataFormat::LocalMax2RelX % wordsize_, 
						    cluster2DDataMap_.at(Cluster2DDataFormat::LocalMax2RelX)));    
  case 1: 
    HGCC.setCluster2DLocalMax1ET(      get32bitPart(DataToBeDecoded.at(Cluster2DDataFormat::LocalMax1ET/wordsize_),   
						    Cluster2DDataFormat::LocalMax1ET % wordsize_,   
						    cluster2DDataMap_.at(Cluster2DDataFormat::LocalMax1ET)));
    HGCC.setCluster2DLocalMax1RelY(    get32bitPart(DataToBeDecoded.at(Cluster2DDataFormat::LocalMax1RelY/wordsize_), 
						    Cluster2DDataFormat::LocalMax1RelY % wordsize_, 
						    cluster2DDataMap_.at(Cluster2DDataFormat::LocalMax1RelY)));
    HGCC.setCluster2DLocalMax1RelX(    get32bitPart(DataToBeDecoded.at(Cluster2DDataFormat::LocalMax1RelX/wordsize_), 
						    Cluster2DDataFormat::LocalMax1RelX % wordsize_, 
						    cluster2DDataMap_.at(Cluster2DDataFormat::LocalMax1RelX)));
  case 0:
    HGCC.setCluster2DLocalMax0ET(      get32bitPart(DataToBeDecoded.at(Cluster2DDataFormat::LocalMax0ET/wordsize_),   
						    Cluster2DDataFormat::LocalMax0ET % wordsize_,   
						    cluster2DDataMap_.at(Cluster2DDataFormat::LocalMax0ET)));
    HGCC.setCluster2DLocalMax0RelY(    get32bitPart(DataToBeDecoded.at(Cluster2DDataFormat::LocalMax0RelX/wordsize_), 
						    Cluster2DDataFormat::LocalMax0RelX % wordsize_,  
						    cluster2DDataMap_.at(Cluster2DDataFormat::LocalMax0RelX)));
  }
  return true; 
}

  
bool HGCalHardwareInterface::encodeCluster2DData (ClusterDataContainer& DataToBeEncoded, const l1t::HGCalCluster & HGCC) const {
  /*  */
  set32bitPart(DataToBeEncoded.at(Cluster2DDataFormat::ET/wordsize_),             
	       HGCC.cluster2DET(),             
	       Cluster2DDataFormat::ET % wordsize_,             
	       cluster2DDataMap_.at(Cluster2DDataFormat::ET));
  set32bitPart(DataToBeEncoded.at(Cluster2DDataFormat::CenterX/wordsize_),        
	       HGCC.cluster2DCenterX(),        
	       Cluster2DDataFormat::CenterX % wordsize_,        
	       cluster2DDataMap_.at(Cluster2DDataFormat::CenterX));
  set32bitPart(DataToBeEncoded.at(Cluster2DDataFormat::CenterY/wordsize_),        
	       HGCC.cluster2DCenterY(),        
	       Cluster2DDataFormat::CenterY % wordsize_,        
	       cluster2DDataMap_.at(Cluster2DDataFormat::CenterY));
  set32bitPart(DataToBeEncoded.at(Cluster2DDataFormat::SizeX/wordsize_),          
	       HGCC.cluster2DSizeX(),          
	       Cluster2DDataFormat::SizeX % wordsize_,          
	       cluster2DDataMap_.at(Cluster2DDataFormat::SizeX));
  set32bitPart(DataToBeEncoded.at(Cluster2DDataFormat::SizeY/wordsize_),          
	       HGCC.cluster2DSizeY(),          
	       Cluster2DDataFormat::SizeY % wordsize_,          
	       cluster2DDataMap_.at(Cluster2DDataFormat::SizeY));
  set32bitPart(DataToBeEncoded.at(Cluster2DDataFormat::NoTriggerCells/wordsize_), 
	       HGCC.cluster2DNoTriggerCells(), 
	       Cluster2DDataFormat::NoTriggerCells % wordsize_, 
	       cluster2DDataMap_.at(Cluster2DDataFormat::NoTriggerCells));
  set32bitPart(DataToBeEncoded.at(Cluster2DDataFormat::LocalMax/wordsize_),       
	       HGCC.cluster2DnLocalMaxima(),       
	       Cluster2DDataFormat::LocalMax % wordsize_,       
	       cluster2DDataMap_.at(Cluster2DDataFormat::LocalMax));
  /* The break statements are omitted since the cases are inclusive in order */ 
  switch ( HGCC.cluster2DnLocalMaxima()){
  case 3: 
    set32bitPart(DataToBeEncoded.at(Cluster2DDataFormat::LocalMax3ET/wordsize_),   
		 HGCC.cluster2DLocalMax3ET(),   
		 Cluster2DDataFormat::LocalMax3ET % wordsize_,   
		 cluster2DDataMap_.at(Cluster2DDataFormat::LocalMax3ET));
    set32bitPart(DataToBeEncoded.at(Cluster2DDataFormat::LocalMax3RelY/wordsize_), 
		 HGCC.cluster2DLocalMax3RelY(), 
		 Cluster2DDataFormat::LocalMax3RelY % wordsize_, 
		 cluster2DDataMap_.at(Cluster2DDataFormat::LocalMax3RelY));
    set32bitPart(DataToBeEncoded.at(Cluster2DDataFormat::LocalMax3RelX/wordsize_), 
		 HGCC.cluster2DLocalMax3RelX(), 
		 Cluster2DDataFormat::LocalMax3RelX % wordsize_, 
		 cluster2DDataMap_.at(Cluster2DDataFormat::LocalMax3RelX));
  case 2:
    set32bitPart(DataToBeEncoded.at(Cluster2DDataFormat::LocalMax2ET/wordsize_),   
		 HGCC.cluster2DLocalMax2ET(),   
		 Cluster2DDataFormat::LocalMax2ET % wordsize_,   
		 cluster2DDataMap_.at(Cluster2DDataFormat::LocalMax2ET));
    set32bitPart(DataToBeEncoded.at(Cluster2DDataFormat::LocalMax2RelY/wordsize_), 
		 HGCC.cluster2DLocalMax2RelY(), 
		 Cluster2DDataFormat::LocalMax2RelY % wordsize_, 
		 cluster2DDataMap_.at(Cluster2DDataFormat::LocalMax2RelY));
    set32bitPart(DataToBeEncoded.at(Cluster2DDataFormat::LocalMax2RelX/wordsize_), 
		 HGCC.cluster2DLocalMax2RelX(), 
		 Cluster2DDataFormat::LocalMax2RelX % wordsize_, 
		 cluster2DDataMap_.at(Cluster2DDataFormat::LocalMax2RelX));    
  case 1: 
    set32bitPart(DataToBeEncoded.at(Cluster2DDataFormat::LocalMax1ET/wordsize_),   
		 HGCC.cluster2DLocalMax1ET(),   
		 Cluster2DDataFormat::LocalMax1ET % wordsize_,   
		 cluster2DDataMap_.at(Cluster2DDataFormat::LocalMax1ET));
    set32bitPart(DataToBeEncoded.at(Cluster2DDataFormat::LocalMax1RelY/wordsize_), 
		 HGCC.cluster2DLocalMax1RelY(), 
		 Cluster2DDataFormat::LocalMax1RelY % wordsize_, 
		 cluster2DDataMap_.at(Cluster2DDataFormat::LocalMax1RelY));
    set32bitPart(DataToBeEncoded.at(Cluster2DDataFormat::LocalMax1RelX/wordsize_), 
		 HGCC.cluster2DLocalMax1RelX(), 
		 Cluster2DDataFormat::LocalMax1RelX % wordsize_, 
		 cluster2DDataMap_.at(Cluster2DDataFormat::LocalMax1RelX));
  case 0:
    set32bitPart(DataToBeEncoded.at(Cluster2DDataFormat::LocalMax0ET/wordsize_),   
		 HGCC.cluster2DLocalMax0ET(),   
		 Cluster2DDataFormat::LocalMax0ET % wordsize_,   
		 cluster2DDataMap_.at(Cluster2DDataFormat::LocalMax0ET));
    set32bitPart(DataToBeEncoded.at(Cluster2DDataFormat::LocalMax0RelY/wordsize_), 
		 HGCC.cluster2DLocalMax0RelY(), 
		 Cluster2DDataFormat::LocalMax0RelY % wordsize_, 
		 cluster2DDataMap_.at(Cluster2DDataFormat::LocalMax0RelY));
    set32bitPart(DataToBeEncoded.at(Cluster2DDataFormat::LocalMax0RelX/wordsize_), 
		 HGCC.cluster2DLocalMax0RelX(), 
		 Cluster2DDataFormat::LocalMax0RelX % wordsize_, 
		 cluster2DDataMap_.at(Cluster2DDataFormat::LocalMax0RelX));
  }
  return true; 
}


bool HGCalHardwareInterface::encodeCluster3DData(ClusterDataContainer& DataToBeEncoded, const l1t::HGCalMulticluster & HGCMC) const {
  /* HGCalMulticluster dataformat encoding */ 
  set32bitPart(DataToBeEncoded.at(Cluster3DDataFormat::ET/wordsize_),             
	       HGCMC.cluster3DET(),             
	       Cluster3DDataFormat::ET % wordsize_,             
	       cluster3DDataMap_.at(Cluster3DDataFormat::ET));
  set32bitPart(DataToBeEncoded.at(Cluster3DDataFormat::Z/wordsize_),              
	       HGCMC.cluster3DZ(),              
	       Cluster3DDataFormat::Z % wordsize_,              
	       cluster3DDataMap_.at(Cluster3DDataFormat::Z));
  set32bitPart(DataToBeEncoded.at(Cluster3DDataFormat::Phi/wordsize_),            
	       HGCMC.cluster3DPhi(),            
	       Cluster3DDataFormat::Phi % wordsize_,            
	       cluster3DDataMap_.at(Cluster3DDataFormat::Phi));
  set32bitPart(DataToBeEncoded.at(Cluster3DDataFormat::Eta/wordsize_),            
	       HGCMC.cluster3DEta(),            
	       Cluster3DDataFormat::Eta % wordsize_,            
	       cluster3DDataMap_.at(Cluster3DDataFormat::Eta));
  set32bitPart(DataToBeEncoded.at(Cluster3DDataFormat::Flags/wordsize_),          
	       HGCMC.cluster3DFlags(),          
	       Cluster3DDataFormat::Flags % wordsize_,          
	       cluster3DDataMap_.at(Cluster3DDataFormat::Flags));
  set32bitPart(DataToBeEncoded.at(Cluster3DDataFormat::QFlags/wordsize_),         
	       HGCMC.cluster3DQFlags(),         
	       Cluster3DDataFormat::QFlags % wordsize_,         
	       cluster3DDataMap_.at(Cluster3DDataFormat::QFlags));
  set32bitPart(DataToBeEncoded.at(Cluster3DDataFormat::NoTriggerCells/wordsize_), 
	       HGCMC.cluster3DNoTriggerCells(), 
	       Cluster3DDataFormat::NoTriggerCells % wordsize_, 
	       cluster3DDataMap_.at(Cluster3DDataFormat::NoTriggerCells));
  set32bitPart(DataToBeEncoded.at(Cluster3DDataFormat::TBA/wordsize_),            
	       HGCMC.cluster3DTBA(),            
	       Cluster3DDataFormat::TBA % wordsize_,            
	       cluster3DDataMap_.at(Cluster3DDataFormat::TBA));  
  set32bitPart(DataToBeEncoded.at(Cluster3DDataFormat::MaxEnergyLayer/wordsize_), 
	       HGCMC.cluster3DMaxEnergyLayer(), 
	       Cluster3DDataFormat::MaxEnergyLayer % wordsize_, 
	       cluster3DDataMap_.at(Cluster3DDataFormat::MaxEnergyLayer));  
  return true;
}


bool HGCalHardwareInterface::decodeCluster3DData (const ClusterDataContainer& DataToBeDecoded, l1t::HGCalMulticluster & HGCMC) const {
  if(DataToBeDecoded.empty()){ 
    return false;
  }
  HGCMC.setCluster3DET(            get32bitPart(DataToBeDecoded.at(Cluster3DDataFormat::ET/wordsize_),             
						Cluster3DDataFormat::ET % wordsize_,             
						cluster3DDataMap_.at(Cluster3DDataFormat::ET)));
  HGCMC.setCluster3DZ(             get32bitPart(DataToBeDecoded.at(Cluster3DDataFormat::Z/wordsize_),        
						Cluster3DDataFormat::Z % wordsize_,        
						cluster3DDataMap_.at(Cluster3DDataFormat::Z)));
  HGCMC.setCluster3DBHEFraction(   get32bitPart(DataToBeDecoded.at(Cluster3DDataFormat::BHEFraction/wordsize_),        
						Cluster3DDataFormat::BHEFraction % wordsize_,        
						cluster3DDataMap_.at(Cluster3DDataFormat::BHEFraction)));
  HGCMC.setCluster3DEEEFraction(   get32bitPart(DataToBeDecoded.at(Cluster3DDataFormat::EEEFraction/wordsize_),          
						Cluster3DDataFormat::EEEFraction % wordsize_,          
						cluster3DDataMap_.at(Cluster3DDataFormat::EEEFraction)));
  HGCMC.setCluster3DPhi(           get32bitPart(DataToBeDecoded.at(Cluster3DDataFormat::Phi/wordsize_),          
						Cluster3DDataFormat::Phi % wordsize_,          
						cluster3DDataMap_.at(Cluster3DDataFormat::Phi)));
  HGCMC.setCluster3DEta(           get32bitPart(DataToBeDecoded.at(Cluster3DDataFormat::Eta/wordsize_),          
						Cluster3DDataFormat::Eta % wordsize_,          
						cluster3DDataMap_.at(Cluster3DDataFormat::Eta)));
  HGCMC.setCluster3DFlags(         get32bitPart(DataToBeDecoded.at(Cluster3DDataFormat::Flags/wordsize_),          
						Cluster3DDataFormat::Flags % wordsize_,          
						cluster3DDataMap_.at(Cluster3DDataFormat::Flags)));
  HGCMC.setCluster3DQFlags(        get32bitPart(DataToBeDecoded.at(Cluster3DDataFormat::QFlags/wordsize_),          
						Cluster3DDataFormat::QFlags % wordsize_,          
						cluster3DDataMap_.at(Cluster3DDataFormat::QFlags)));
  HGCMC.setCluster3DNoTriggerCells(get32bitPart(DataToBeDecoded.at(Cluster3DDataFormat::NoTriggerCells/wordsize_), 
						Cluster3DDataFormat::NoTriggerCells % wordsize_, 
						cluster3DDataMap_.at(Cluster3DDataFormat::NoTriggerCells)));
  HGCMC.setCluster3DTBA(           get32bitPart(DataToBeDecoded.at(Cluster3DDataFormat::TBA/wordsize_), 
						Cluster3DDataFormat::TBA % wordsize_, 
						cluster3DDataMap_.at(Cluster3DDataFormat::TBA)));
  HGCMC.setCluster3DMaxEnergyLayer(get32bitPart(DataToBeDecoded.at(Cluster3DDataFormat::MaxEnergyLayer/wordsize_),       
						Cluster3DDataFormat::MaxEnergyLayer % wordsize_,       
						cluster3DDataMap_.at(Cluster3DDataFormat::MaxEnergyLayer)));
  return true; 
}


const HGCalHardwareInterface::DataFormatMap  HGCalHardwareInterface::cluster2DDataMap_ =  {
  /*Initalize dataformat map: bit-size for each word */  
  {Cluster2DDataFormat::ET            , 8},
  {Cluster2DDataFormat::CenterY       , 12},
  {Cluster2DDataFormat::CenterX       , 12},
  {Cluster2DDataFormat::Flags         , 6},
  {Cluster2DDataFormat::SizeY         , 8},
  {Cluster2DDataFormat::SizeX         , 8},
  {Cluster2DDataFormat::LocalMax      , 2},
  {Cluster2DDataFormat::NoTriggerCells, 8},
  {Cluster2DDataFormat::LocalMax0RelY , 8},  
  {Cluster2DDataFormat::LocalMax0RelX , 8},
  {Cluster2DDataFormat::LocalMax0ET   , 8},  
  {Cluster2DDataFormat::LocalMax1RelY , 8},  
  {Cluster2DDataFormat::LocalMax1RelX , 8},
  {Cluster2DDataFormat::LocalMax1ET   , 8},  
  {Cluster2DDataFormat::LocalMax2RelY , 8},  
  {Cluster2DDataFormat::LocalMax2RelX , 8},
  {Cluster2DDataFormat::LocalMax2ET   , 8},  
  {Cluster2DDataFormat::LocalMax3RelY , 8},  
  {Cluster2DDataFormat::LocalMax3RelX , 8},
  {Cluster2DDataFormat::LocalMax3ET   , 8}};

const HGCalHardwareInterface::DataFormatMap  HGCalHardwareInterface::cluster3DDataMap_ =  {
  {Cluster3DDataFormat::BHEFraction    , 8},
  {Cluster3DDataFormat::EEEFraction    , 8},
  {Cluster3DDataFormat::ET             , 16},
  {Cluster3DDataFormat::Z              , 10},
  {Cluster3DDataFormat::Phi            , 11},
  {Cluster3DDataFormat::Eta            , 11},
  {Cluster3DDataFormat::Flags          , 12},
  {Cluster3DDataFormat::QFlags         , 12},
  {Cluster3DDataFormat::NoTriggerCells , 8},
  {Cluster3DDataFormat::TBA            , 26},
  {Cluster3DDataFormat::MaxEnergyLayer , 6}
};
   
short  HGCalHardwareInterface::get32bitPart  (const std::uint32_t data32bit, const unsigned short lsb, const unsigned short size ) const{
  return (data32bit >> lsb) & (0xffffffff >> (wordsize_-size));
}

void  HGCalHardwareInterface::del32bitPart (std::uint32_t & data32bit,  const unsigned short lsb, const unsigned short size ) const{
  data32bit = data32bit &  (((lsb > 0) ? (0xffffffff  >> (wordsize_-lsb)) : 0x0 ) | ((lsb+size > wordsize_-1)? 0x0 : 0xffffffff  << (lsb+size)));
}

void  HGCalHardwareInterface::set32bitPart  (std::uint32_t & data32bit, const std::int32_t bitvalue,  const unsigned short lsb, const unsigned short size ) const{
  if( bitvalue > std::pow(2, size) - 1 || -1 * bitvalue > std::pow(2,size-1) ){
    throw cms::Exception("the value exceeds the given bit range");
  }
  if(size > wordsize_ || lsb > wordsize_-1 ){
    throw cms::Exception("the value exceeds 32-bit");
  }
  del32bitPart(data32bit,lsb, size);
  data32bit = data32bit | (bitvalue << lsb);
}
