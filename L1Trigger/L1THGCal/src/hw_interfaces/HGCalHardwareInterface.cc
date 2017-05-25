#include "L1Trigger/L1THGCal/interface/hw_interfaces/HGCalHardwareInterface.h"

bool HGCalHardwareInterface::decodeClusterData (const ClusterDataContainer& DataToBeDecoded, l1t::HGCalCluster & HGCC) const{
  if(DataToBeDecoded.empty()) {
    return false;
  }
  HGCC.setCluster2DET(            get32bitPart(DataToBeDecoded.at(ClusterDataFormat::ET/wordsize_),             
					       ClusterDataFormat::ET % wordsize_,             
					       clusterDataMap_.at(ClusterDataFormat::ET)));
  HGCC.setCluster2DCenterX(       get32bitPart(DataToBeDecoded.at(ClusterDataFormat::CenterX/wordsize_),        
					       ClusterDataFormat::CenterX % wordsize_,        
					       clusterDataMap_.at(ClusterDataFormat::CenterX)));
  HGCC.setCluster2DCenterY(       get32bitPart(DataToBeDecoded.at(ClusterDataFormat::CenterY/wordsize_),        
					       ClusterDataFormat::CenterY % wordsize_,        
					       clusterDataMap_.at(ClusterDataFormat::CenterY)));
  HGCC.setCluster2DSizeX(         get32bitPart(DataToBeDecoded.at(ClusterDataFormat::SizeX/wordsize_),          
					       ClusterDataFormat::SizeX % wordsize_,          
					       clusterDataMap_.at(ClusterDataFormat::SizeX)));
  HGCC.setCluster2DSizeY(         get32bitPart(DataToBeDecoded.at(ClusterDataFormat::SizeY/wordsize_),          
					       ClusterDataFormat::SizeY % wordsize_,          
					       clusterDataMap_.at(ClusterDataFormat::SizeY)));
  HGCC.setCluster2DNoTriggerCells(get32bitPart(DataToBeDecoded.at(ClusterDataFormat::NoTriggerCells/wordsize_), 
					       ClusterDataFormat::NoTriggerCells % wordsize_, 
					       clusterDataMap_.at(ClusterDataFormat::NoTriggerCells)));
  HGCC.setCluster2DnLocalMaxima(  get32bitPart(DataToBeDecoded.at(ClusterDataFormat::LocalMax/wordsize_),       
					       ClusterDataFormat::LocalMax % wordsize_,       
					       clusterDataMap_.at(ClusterDataFormat::LocalMax)));
  /* The break statements are omitted since the cases are inclusive in order */ 
  switch ( HGCC.cluster2DnLocalMaxima()){
  case 3: 
    HGCC.setCluster2DLocalMax3ET(      get32bitPart(DataToBeDecoded.at(ClusterDataFormat::LocalMax3ET/wordsize_),   
						    ClusterDataFormat::LocalMax3ET % wordsize_,   
						    clusterDataMap_.at(ClusterDataFormat::LocalMax3ET)));
    HGCC.setCluster2DLocalMax3RelY(    get32bitPart(DataToBeDecoded.at(ClusterDataFormat::LocalMax3RelY/wordsize_), 
						    ClusterDataFormat::LocalMax3RelY % wordsize_, 
						    clusterDataMap_.at(ClusterDataFormat::LocalMax3RelY)));
    HGCC.setCluster2DLocalMax3RelX(    get32bitPart(DataToBeDecoded.at(ClusterDataFormat::LocalMax3RelX/wordsize_), 
						    ClusterDataFormat::LocalMax3RelX % wordsize_, 
						    clusterDataMap_.at(ClusterDataFormat::LocalMax3RelX)));
  case 2:
    HGCC.setCluster2DLocalMax2ET(      get32bitPart(DataToBeDecoded.at(ClusterDataFormat::LocalMax2ET/wordsize_),   
						    ClusterDataFormat::LocalMax2ET % wordsize_,   
						    clusterDataMap_.at(ClusterDataFormat::LocalMax2ET)));
    HGCC.setCluster2DLocalMax2RelY(    get32bitPart(DataToBeDecoded.at(ClusterDataFormat::LocalMax2RelY/wordsize_), 
						    ClusterDataFormat::LocalMax2RelY % wordsize_, 
						    clusterDataMap_.at(ClusterDataFormat::LocalMax2RelY)));
    HGCC.setCluster2DLocalMax2RelX(    get32bitPart(DataToBeDecoded.at(ClusterDataFormat::LocalMax2RelX/wordsize_), 
						    ClusterDataFormat::LocalMax2RelX % wordsize_, 
						    clusterDataMap_.at(ClusterDataFormat::LocalMax2RelX)));    
  case 1: 
    HGCC.setCluster2DLocalMax1ET(      get32bitPart(DataToBeDecoded.at(ClusterDataFormat::LocalMax1ET/wordsize_),   
						    ClusterDataFormat::LocalMax1ET % wordsize_,   
						    clusterDataMap_.at(ClusterDataFormat::LocalMax1ET)));
    HGCC.setCluster2DLocalMax1RelY(    get32bitPart(DataToBeDecoded.at(ClusterDataFormat::LocalMax1RelY/wordsize_), 
						    ClusterDataFormat::LocalMax1RelY % wordsize_, 
						    clusterDataMap_.at(ClusterDataFormat::LocalMax1RelY)));
    HGCC.setCluster2DLocalMax1RelX(    get32bitPart(DataToBeDecoded.at(ClusterDataFormat::LocalMax1RelX/wordsize_), 
						    ClusterDataFormat::LocalMax1RelX % wordsize_, 
						    clusterDataMap_.at(ClusterDataFormat::LocalMax1RelX)));
  case 0:
    HGCC.setCluster2DLocalMax0ET(      get32bitPart(DataToBeDecoded.at(ClusterDataFormat::LocalMax0ET/wordsize_),   
						    ClusterDataFormat::LocalMax0ET % wordsize_,   
						    clusterDataMap_.at(ClusterDataFormat::LocalMax0ET)));
    HGCC.setCluster2DLocalMax0RelY(    get32bitPart(DataToBeDecoded.at(ClusterDataFormat::LocalMax0RelY/wordsize_), 
						    ClusterDataFormat::LocalMax0RelY % wordsize_, 
						    clusterDataMap_.at(ClusterDataFormat::LocalMax0RelY)));
    HGCC.setCluster2DLocalMax0RelX(    get32bitPart(DataToBeDecoded.at(ClusterDataFormat::LocalMax0RelX/wordsize_), 
						    ClusterDataFormat::LocalMax0RelX % wordsize_,  
						    clusterDataMap_.at(ClusterDataFormat::LocalMax0RelX)));
  }
  return true; 
}


bool HGCalHardwareInterface::encodeClusterData (ClusterDataContainer& DataToBeEncoded, const l1t::HGCalCluster & HGCC) const{
  /*  */
  set32bitPart(DataToBeEncoded.at(ClusterDataFormat::ET/wordsize_),             
	       HGCC.cluster2DET(),             
	       ClusterDataFormat::ET % wordsize_,             
	       clusterDataMap_.at(ClusterDataFormat::ET));
  set32bitPart(DataToBeEncoded.at(ClusterDataFormat::CenterX/wordsize_),        
	       HGCC.cluster2DCenterX(),        
	       ClusterDataFormat::CenterX % wordsize_,        
	       clusterDataMap_.at(ClusterDataFormat::CenterX));
  set32bitPart(DataToBeEncoded.at(ClusterDataFormat::CenterY/wordsize_),        
	       HGCC.cluster2DCenterY(),        
	       ClusterDataFormat::CenterY % wordsize_,        
	       clusterDataMap_.at(ClusterDataFormat::CenterY));
  set32bitPart(DataToBeEncoded.at(ClusterDataFormat::SizeX/wordsize_),          
	       HGCC.cluster2DSizeX(),          
	       ClusterDataFormat::SizeX % wordsize_,          
	       clusterDataMap_.at(ClusterDataFormat::SizeX));
  set32bitPart(DataToBeEncoded.at(ClusterDataFormat::SizeY/wordsize_),          
	       HGCC.cluster2DSizeY(),          
	       ClusterDataFormat::SizeY % wordsize_,          
	       clusterDataMap_.at(ClusterDataFormat::SizeY));
  set32bitPart(DataToBeEncoded.at(ClusterDataFormat::NoTriggerCells/wordsize_), 
	       HGCC.cluster2DNoTriggerCells(), 
	       ClusterDataFormat::NoTriggerCells % wordsize_, 
	       clusterDataMap_.at(ClusterDataFormat::NoTriggerCells));
  set32bitPart(DataToBeEncoded.at(ClusterDataFormat::LocalMax/wordsize_),       
	       HGCC.cluster2DnLocalMaxima(),       
	       ClusterDataFormat::LocalMax % wordsize_,       
	       clusterDataMap_.at(ClusterDataFormat::LocalMax));
  /* The break statements are omitted since the cases are inclusive in order */ 
  switch ( HGCC.cluster2DnLocalMaxima()){
  case 3: 
    set32bitPart(DataToBeEncoded.at(ClusterDataFormat::LocalMax3ET/wordsize_),   
		 HGCC.cluster2DLocalMax3ET(),   
		 ClusterDataFormat::LocalMax3ET % wordsize_,   
		 clusterDataMap_.at(ClusterDataFormat::LocalMax3ET));
    set32bitPart(DataToBeEncoded.at(ClusterDataFormat::LocalMax3RelY/wordsize_), 
		 HGCC.cluster2DLocalMax3RelY(), 
		 ClusterDataFormat::LocalMax3RelY % wordsize_, 
		 clusterDataMap_.at(ClusterDataFormat::LocalMax3RelY));
    set32bitPart(DataToBeEncoded.at(ClusterDataFormat::LocalMax3RelX/wordsize_), 
		 HGCC.cluster2DLocalMax3RelX(), 
		 ClusterDataFormat::LocalMax3RelX % wordsize_, 
		 clusterDataMap_.at(ClusterDataFormat::LocalMax3RelX));
  case 2:
    set32bitPart(DataToBeEncoded.at(ClusterDataFormat::LocalMax2ET/wordsize_),   
		 HGCC.cluster2DLocalMax2ET(),   
		 ClusterDataFormat::LocalMax2ET % wordsize_,   
		 clusterDataMap_.at(ClusterDataFormat::LocalMax2ET));
    set32bitPart(DataToBeEncoded.at(ClusterDataFormat::LocalMax2RelY/wordsize_), 
		 HGCC.cluster2DLocalMax2RelY(), 
		 ClusterDataFormat::LocalMax2RelY % wordsize_, 
		 clusterDataMap_.at(ClusterDataFormat::LocalMax2RelY));
    set32bitPart(DataToBeEncoded.at(ClusterDataFormat::LocalMax2RelX/wordsize_), 
		 HGCC.cluster2DLocalMax2RelX(), 
		 ClusterDataFormat::LocalMax2RelX % wordsize_, 
		 clusterDataMap_.at(ClusterDataFormat::LocalMax2RelX));    
  case 1: 
    set32bitPart(DataToBeEncoded.at(ClusterDataFormat::LocalMax1ET/wordsize_),   
		 HGCC.cluster2DLocalMax1ET(),   
		 ClusterDataFormat::LocalMax1ET % wordsize_,   
		 clusterDataMap_.at(ClusterDataFormat::LocalMax1ET));
    set32bitPart(DataToBeEncoded.at(ClusterDataFormat::LocalMax1RelY/wordsize_), 
		 HGCC.cluster2DLocalMax1RelY(), 
		 ClusterDataFormat::LocalMax1RelY % wordsize_, 
		 clusterDataMap_.at(ClusterDataFormat::LocalMax1RelY));
    set32bitPart(DataToBeEncoded.at(ClusterDataFormat::LocalMax1RelX/wordsize_), 
		 HGCC.cluster2DLocalMax1RelX(), 
		 ClusterDataFormat::LocalMax1RelX % wordsize_, 
		 clusterDataMap_.at(ClusterDataFormat::LocalMax1RelX));
  case 0:
    set32bitPart(DataToBeEncoded.at(ClusterDataFormat::LocalMax0ET/wordsize_),   
		 HGCC.cluster2DLocalMax0ET(),   
		 ClusterDataFormat::LocalMax0ET % wordsize_,   
		 clusterDataMap_.at(ClusterDataFormat::LocalMax0ET));
    set32bitPart(DataToBeEncoded.at(ClusterDataFormat::LocalMax0RelY/wordsize_), 
		 HGCC.cluster2DLocalMax0RelY(), 
		 ClusterDataFormat::LocalMax0RelY % wordsize_, 
		 clusterDataMap_.at(ClusterDataFormat::LocalMax0RelY));
    set32bitPart(DataToBeEncoded.at(ClusterDataFormat::LocalMax0RelX/wordsize_), 
		 HGCC.cluster2DLocalMax0RelX(), 
		 ClusterDataFormat::LocalMax0RelX % wordsize_, 
		 clusterDataMap_.at(ClusterDataFormat::LocalMax0RelX));
  }
  return true; 
}


bool HGCalHardwareInterface::encodeMulticlusterData(ClusterDataContainer& DataToBeEncoded, const l1t::HGCalMulticluster & HGCMC) const {
  /* HGCalMulticluster dataformat encoding */ 
  set32bitPart(DataToBeEncoded.at(MulticlusterDataFormat::ET/wordsize_),             
	       HGCMC.cluster3DET(),             
	       MulticlusterDataFormat::ET % wordsize_,             
	       multiclusterDataMap_.at(MulticlusterDataFormat::ET));
  set32bitPart(DataToBeEncoded.at(MulticlusterDataFormat::Z/wordsize_),              
	       HGCMC.cluster3DZ(),              
	       MulticlusterDataFormat::Z % wordsize_,              
	       multiclusterDataMap_.at(MulticlusterDataFormat::Z));
  set32bitPart(DataToBeEncoded.at(MulticlusterDataFormat::Phi/wordsize_),            
	       HGCMC.cluster3DPhi(),            
	       MulticlusterDataFormat::Phi % wordsize_,            
	       multiclusterDataMap_.at(MulticlusterDataFormat::Phi));
  set32bitPart(DataToBeEncoded.at(MulticlusterDataFormat::Eta/wordsize_),            
	       HGCMC.cluster3DEta(),            
	       MulticlusterDataFormat::Eta % wordsize_,            
	       multiclusterDataMap_.at(MulticlusterDataFormat::Eta));
  set32bitPart(DataToBeEncoded.at(MulticlusterDataFormat::Flags/wordsize_),          
	       HGCMC.cluster3DFlags(),          
	       MulticlusterDataFormat::Flags % wordsize_,          
	       multiclusterDataMap_.at(MulticlusterDataFormat::Flags));
  set32bitPart(DataToBeEncoded.at(MulticlusterDataFormat::QFlags/wordsize_),         
	       HGCMC.cluster3DQFlags(),         
	       MulticlusterDataFormat::QFlags % wordsize_,         
	       multiclusterDataMap_.at(MulticlusterDataFormat::QFlags));
  set32bitPart(DataToBeEncoded.at(MulticlusterDataFormat::NoTriggerCells/wordsize_), 
	       HGCMC.cluster3DNoTriggerCells(), 
	       MulticlusterDataFormat::NoTriggerCells % wordsize_, 
	       multiclusterDataMap_.at(MulticlusterDataFormat::NoTriggerCells));
  set32bitPart(DataToBeEncoded.at(MulticlusterDataFormat::TBA/wordsize_),            
	       HGCMC.cluster3DTBA(),            
	       MulticlusterDataFormat::TBA % wordsize_,            
	       multiclusterDataMap_.at(MulticlusterDataFormat::TBA));  
  set32bitPart(DataToBeEncoded.at(MulticlusterDataFormat::MaxEnergyLayer/wordsize_), 
	       HGCMC.cluster3DMaxEnergyLayer(), 
	       MulticlusterDataFormat::MaxEnergyLayer % wordsize_, 
	       multiclusterDataMap_.at(MulticlusterDataFormat::MaxEnergyLayer));  
  return true;
}


bool HGCalHardwareInterface::decodeMulticlusterData (const ClusterDataContainer& DataToBeDecoded, l1t::HGCalMulticluster & HGCMC) const {
  if(DataToBeDecoded.empty()){ 
    return false;
  }
  HGCMC.setCluster3DET(            get32bitPart(DataToBeDecoded.at(MulticlusterDataFormat::ET/wordsize_),             
						MulticlusterDataFormat::ET % wordsize_,             
						multiclusterDataMap_.at(MulticlusterDataFormat::ET)));
  HGCMC.setCluster3DZ(             get32bitPart(DataToBeDecoded.at(MulticlusterDataFormat::Z/wordsize_),        
						MulticlusterDataFormat::Z % wordsize_,        
						multiclusterDataMap_.at(MulticlusterDataFormat::Z)));
  HGCMC.setCluster3DBHEFraction(   get32bitPart(DataToBeDecoded.at(MulticlusterDataFormat::BHEFraction/wordsize_),        
						MulticlusterDataFormat::BHEFraction % wordsize_,        
						multiclusterDataMap_.at(MulticlusterDataFormat::BHEFraction)));
  HGCMC.setCluster3DEEEFraction(   get32bitPart(DataToBeDecoded.at(MulticlusterDataFormat::EEEFraction/wordsize_),          
						MulticlusterDataFormat::EEEFraction % wordsize_,          
						multiclusterDataMap_.at(MulticlusterDataFormat::EEEFraction)));
  HGCMC.setCluster3DPhi(           get32bitPart(DataToBeDecoded.at(MulticlusterDataFormat::Phi/wordsize_),          
						MulticlusterDataFormat::Phi % wordsize_,          
						multiclusterDataMap_.at(MulticlusterDataFormat::Phi)));
  HGCMC.setCluster3DEta(           get32bitPart(DataToBeDecoded.at(MulticlusterDataFormat::Eta/wordsize_),          
						MulticlusterDataFormat::Eta % wordsize_,          
						multiclusterDataMap_.at(MulticlusterDataFormat::Eta)));
  HGCMC.setCluster3DFlags(         get32bitPart(DataToBeDecoded.at(MulticlusterDataFormat::Flags/wordsize_),          
						MulticlusterDataFormat::Flags % wordsize_,          
						multiclusterDataMap_.at(MulticlusterDataFormat::Flags)));
  HGCMC.setCluster3DQFlags(        get32bitPart(DataToBeDecoded.at(MulticlusterDataFormat::QFlags/wordsize_),          
						MulticlusterDataFormat::QFlags % wordsize_,          
						multiclusterDataMap_.at(MulticlusterDataFormat::QFlags)));
  HGCMC.setCluster3DNoTriggerCells(get32bitPart(DataToBeDecoded.at(MulticlusterDataFormat::NoTriggerCells/wordsize_), 
						MulticlusterDataFormat::NoTriggerCells % wordsize_, 
						multiclusterDataMap_.at(MulticlusterDataFormat::NoTriggerCells)));
  HGCMC.setCluster3DTBA(           get32bitPart(DataToBeDecoded.at(MulticlusterDataFormat::TBA/wordsize_), 
						MulticlusterDataFormat::TBA % wordsize_, 
						multiclusterDataMap_.at(MulticlusterDataFormat::TBA)));
  HGCMC.setCluster3DMaxEnergyLayer(get32bitPart(DataToBeDecoded.at(MulticlusterDataFormat::MaxEnergyLayer/wordsize_),       
						MulticlusterDataFormat::MaxEnergyLayer % wordsize_,       
						multiclusterDataMap_.at(MulticlusterDataFormat::MaxEnergyLayer)));
  return true; 
}


const HGCalHardwareInterface::DataFormatMap  HGCalHardwareInterface::clusterDataMap_ =  {
  /*Initalize dataformat map: bit-size for each word */  
  {ClusterDataFormat::ET            , 8},
  {ClusterDataFormat::CenterY       , 12},
  {ClusterDataFormat::CenterX       , 12},
  {ClusterDataFormat::Flags         , 6},
  {ClusterDataFormat::SizeY         , 8},
  {ClusterDataFormat::SizeX         , 8},
  {ClusterDataFormat::LocalMax      , 2},
  {ClusterDataFormat::NoTriggerCells, 8},
  {ClusterDataFormat::LocalMax0RelY , 8},  
  {ClusterDataFormat::LocalMax0RelX , 8},
  {ClusterDataFormat::LocalMax0ET   , 8},  
  {ClusterDataFormat::LocalMax1RelY , 8},  
  {ClusterDataFormat::LocalMax1RelX , 8},
  {ClusterDataFormat::LocalMax1ET   , 8},  
  {ClusterDataFormat::LocalMax2RelY , 8},  
  {ClusterDataFormat::LocalMax2RelX , 8},
  {ClusterDataFormat::LocalMax2ET   , 8},  
  {ClusterDataFormat::LocalMax3RelY , 8},  
  {ClusterDataFormat::LocalMax3RelX , 8},
  {ClusterDataFormat::LocalMax3ET   , 8}};

const HGCalHardwareInterface::DataFormatMap  HGCalHardwareInterface::multiclusterDataMap_ =  {
  {MulticlusterDataFormat::BHEFraction    , 8},
  {MulticlusterDataFormat::EEEFraction    , 8},
  {MulticlusterDataFormat::ET             , 16},
  {MulticlusterDataFormat::Z              , 10},
  {MulticlusterDataFormat::Phi            , 11},
  {MulticlusterDataFormat::Eta            , 11},
  {MulticlusterDataFormat::Flags          , 12},
  {MulticlusterDataFormat::QFlags         , 12},
  {MulticlusterDataFormat::NoTriggerCells , 8},
  {MulticlusterDataFormat::TBA            , 26},
  {MulticlusterDataFormat::MaxEnergyLayer , 6}
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
