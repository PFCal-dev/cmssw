#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalHardwareInterface.h"


bool HGCalHardwareInterface::DecodeClusterData (const ClusterDataContainer& DataToBeDecoded){
  if(DataToBeDecoded.empty()) 
    return false;
  HGCMC_->SetClusterET(            get32bitPart_(DataToBeDecoded.at(ClusterDataFormat::ET/wordsize_),             
						 ClusterDataFormat::ET % wordsize_,             
						 ClusterDataMap_[ClusterDataFormat::ET]));
  HGCMC_->SetClusterCenterX(       get32bitPart_(DataToBeDecoded.at(ClusterDataFormat::CenterX/wordsize_),        
						 ClusterDataFormat::CenterX % wordsize_,        
						 ClusterDataMap_[ClusterDataFormat::CenterX]));
  HGCMC_->SetClusterCenterY(       get32bitPart_(DataToBeDecoded.at(ClusterDataFormat::CenterY/wordsize_),        
						 ClusterDataFormat::CenterY % wordsize_,        
						 ClusterDataMap_[ClusterDataFormat::CenterY]));
  HGCMC_->SetClusterSizeX(         get32bitPart_(DataToBeDecoded.at(ClusterDataFormat::SizeX/wordsize_),          
						 ClusterDataFormat::SizeX % wordsize_,          
						 ClusterDataMap_[ClusterDataFormat::SizeX]));
  HGCMC_->SetClusterSizeY(         get32bitPart_(DataToBeDecoded.at(ClusterDataFormat::SizeY/wordsize_),          
						 ClusterDataFormat::SizeY % wordsize_,          
						 ClusterDataMap_[ClusterDataFormat::SizeY]));
  HGCMC_->SetClusterNoTriggerCells(get32bitPart_(DataToBeDecoded.at(ClusterDataFormat::NoTriggerCells/wordsize_), 
						 ClusterDataFormat::NoTriggerCells % wordsize_, 
						 ClusterDataMap_[ClusterDataFormat::NoTriggerCells]));
  HGCMC_->SetClusterLocalMax(      get32bitPart_(DataToBeDecoded.at(ClusterDataFormat::LocalMax/wordsize_),       
						 ClusterDataFormat::LocalMax % wordsize_,       
						 ClusterDataMap_[ClusterDataFormat::LocalMax]));
  switch ( HGCMC_->ClusterLocalMax()){
  case 3: 
    HGCMC_->SetClusterLocalMax3ET(      get32bitPart_(DataToBeDecoded.at(ClusterDataFormat::LocalMax3ET/wordsize_),   
						      ClusterDataFormat::LocalMax3ET % wordsize_,   
						      ClusterDataMap_[ClusterDataFormat::LocalMax3ET]));
    HGCMC_->SetClusterLocalMax3RelY(    get32bitPart_(DataToBeDecoded.at(ClusterDataFormat::LocalMax3RelY/wordsize_), 
						      ClusterDataFormat::LocalMax3RelY % wordsize_, 
						      ClusterDataMap_[ClusterDataFormat::LocalMax3RelY]));
    HGCMC_->SetClusterLocalMax3RelX(    get32bitPart_(DataToBeDecoded.at(ClusterDataFormat::LocalMax3RelX/wordsize_), 
						      ClusterDataFormat::LocalMax3RelX % wordsize_, 
						      ClusterDataMap_[ClusterDataFormat::LocalMax3RelX]));
  case 2:
    HGCMC_->SetClusterLocalMax2ET(      get32bitPart_(DataToBeDecoded.at(ClusterDataFormat::LocalMax2ET/wordsize_),   
						      ClusterDataFormat::LocalMax2ET % wordsize_,   
						      ClusterDataMap_[ClusterDataFormat::LocalMax2ET]));
    HGCMC_->SetClusterLocalMax2RelY(    get32bitPart_(DataToBeDecoded.at(ClusterDataFormat::LocalMax2RelY/wordsize_), 
						      ClusterDataFormat::LocalMax2RelY % wordsize_, 
						      ClusterDataMap_[ClusterDataFormat::LocalMax2RelY]));
    HGCMC_->SetClusterLocalMax2RelX(    get32bitPart_(DataToBeDecoded.at(ClusterDataFormat::LocalMax2RelX/wordsize_), 
						      ClusterDataFormat::LocalMax2RelX % wordsize_, 
						      ClusterDataMap_[ClusterDataFormat::LocalMax2RelX]));    
  case 1: 
    HGCMC_->SetClusterLocalMax1ET(      get32bitPart_(DataToBeDecoded.at(ClusterDataFormat::LocalMax1ET/wordsize_),   
						      ClusterDataFormat::LocalMax1ET % wordsize_,   
						      ClusterDataMap_[ClusterDataFormat::LocalMax1ET]));
    HGCMC_->SetClusterLocalMax1RelY(    get32bitPart_(DataToBeDecoded.at(ClusterDataFormat::LocalMax1RelY/wordsize_), 
						      ClusterDataFormat::LocalMax1RelY % wordsize_, 
						      ClusterDataMap_[ClusterDataFormat::LocalMax1RelY]));
    HGCMC_->SetClusterLocalMax1RelX(    get32bitPart_(DataToBeDecoded.at(ClusterDataFormat::LocalMax1RelX/wordsize_), 
						      ClusterDataFormat::LocalMax1RelX % wordsize_, 
						      ClusterDataMap_[ClusterDataFormat::LocalMax1RelX]));
  case 0:
    HGCMC_->SetClusterLocalMax0ET(      get32bitPart_(DataToBeDecoded.at(ClusterDataFormat::LocalMax0ET/wordsize_),   
						      ClusterDataFormat::LocalMax0ET % wordsize_,   
						      ClusterDataMap_[ClusterDataFormat::LocalMax0ET]));
    HGCMC_->SetClusterLocalMax0RelY(    get32bitPart_(DataToBeDecoded.at(ClusterDataFormat::LocalMax0RelY/wordsize_), 
						      ClusterDataFormat::LocalMax0RelY % wordsize_, 
						      ClusterDataMap_[ClusterDataFormat::LocalMax0RelY]));
    HGCMC_->SetClusterLocalMax0RelX(    get32bitPart_(DataToBeDecoded.at(ClusterDataFormat::LocalMax0RelX/wordsize_), 
						      ClusterDataFormat::LocalMax0RelX % wordsize_,  
						      ClusterDataMap_[ClusterDataFormat::LocalMax0RelX]));
  }
  return true; 
}


bool HGCalHardwareInterface::EncodeClusterData (ClusterDataContainer& DataToBeEncoded){
  /**/
  set32bitPart_(DataToBeEncoded.at(ClusterDataFormat::ET/wordsize_),             HGCC_->ClusterET(),             ClusterDataFormat::ET % wordsize_,             ClusterDataMap_[ClusterDataFormat::ET]);
  set32bitPart_(DataToBeEncoded.at(ClusterDataFormat::CenterX/wordsize_),        HGCC_->ClusterCenterX(),        ClusterDataFormat::CenterX % wordsize_,        ClusterDataMap_[ClusterDataFormat::CenterX]);
  set32bitPart_(DataToBeEncoded.at(ClusterDataFormat::CenterY/wordsize_),        HGCC_->ClusterCenterY(),        ClusterDataFormat::CenterY % wordsize_,        ClusterDataMap_[ClusterDataFormat::CenterY]);
  set32bitPart_(DataToBeEncoded.at(ClusterDataFormat::SizeX/wordsize_),          HGCC_->ClusterSizeX(),          ClusterDataFormat::SizeX % wordsize_,          ClusterDataMap_[ClusterDataFormat::SizeX]);
  set32bitPart_(DataToBeEncoded.at(ClusterDataFormat::SizeY/wordsize_),          HGCC_->ClusterSizeY(),          ClusterDataFormat::SizeY % wordsize_,          ClusterDataMap_[ClusterDataFormat::SizeY]);
  set32bitPart_(DataToBeEncoded.at(ClusterDataFormat::NoTriggerCells/wordsize_), HGCC_->ClusterNoTriggerCells(), ClusterDataFormat::NoTriggerCells % wordsize_, ClusterDataMap_[ClusterDataFormat::NoTriggerCells]);
  set32bitPart_(DataToBeEncoded.at(ClusterDataFormat::LocalMax/wordsize_),       HGCC_->ClusterLocalMax(),       ClusterDataFormat::LocalMax % wordsize_,       ClusterDataMap_[ClusterDataFormat::LocalMax]);
  switch ( HGCC_->ClusterLocalMax()){
  case 3: 
    set32bitPart_(DataToBeEncoded.at(ClusterDataFormat::LocalMax3ET/wordsize_),   HGCC_->ClusterLocalMax3ET(),   ClusterDataFormat::LocalMax3ET % wordsize_,   ClusterDataMap_[ClusterDataFormat::LocalMax3ET]);
    set32bitPart_(DataToBeEncoded.at(ClusterDataFormat::LocalMax3RelY/wordsize_), HGCC_->ClusterLocalMax3RelY(), ClusterDataFormat::LocalMax3RelY % wordsize_, ClusterDataMap_[ClusterDataFormat::LocalMax3RelY]);
    set32bitPart_(DataToBeEncoded.at(ClusterDataFormat::LocalMax3RelX/wordsize_), HGCC_->ClusterLocalMax3RelX(), ClusterDataFormat::LocalMax3RelX % wordsize_, ClusterDataMap_[ClusterDataFormat::LocalMax3RelX]);
  case 2:
    set32bitPart_(DataToBeEncoded.at(ClusterDataFormat::LocalMax2ET/wordsize_),   HGCC_->ClusterLocalMax2ET(),   ClusterDataFormat::LocalMax2ET % wordsize_,   ClusterDataMap_[ClusterDataFormat::LocalMax2ET]);
    set32bitPart_(DataToBeEncoded.at(ClusterDataFormat::LocalMax2RelY/wordsize_), HGCC_->ClusterLocalMax2RelY(), ClusterDataFormat::LocalMax2RelY % wordsize_, ClusterDataMap_[ClusterDataFormat::LocalMax2RelY]);
    set32bitPart_(DataToBeEncoded.at(ClusterDataFormat::LocalMax2RelX/wordsize_), HGCC_->ClusterLocalMax2RelX(), ClusterDataFormat::LocalMax2RelX % wordsize_, ClusterDataMap_[ClusterDataFormat::LocalMax2RelX]);    
  case 1: 
    set32bitPart_(DataToBeEncoded.at(ClusterDataFormat::LocalMax1ET/wordsize_),   HGCC_->ClusterLocalMax1ET(),   ClusterDataFormat::LocalMax1ET % wordsize_,   ClusterDataMap_[ClusterDataFormat::LocalMax1ET]);
    set32bitPart_(DataToBeEncoded.at(ClusterDataFormat::LocalMax1RelY/wordsize_), HGCC_->ClusterLocalMax1RelY(), ClusterDataFormat::LocalMax1RelY % wordsize_, ClusterDataMap_[ClusterDataFormat::LocalMax1RelY]);
    set32bitPart_(DataToBeEncoded.at(ClusterDataFormat::LocalMax1RelX/wordsize_), HGCC_->ClusterLocalMax1RelX(), ClusterDataFormat::LocalMax1RelX % wordsize_, ClusterDataMap_[ClusterDataFormat::LocalMax1RelX]);
  case 0:
    set32bitPart_(DataToBeEncoded.at(ClusterDataFormat::LocalMax0ET/wordsize_),   HGCC_->ClusterLocalMax0ET(),   ClusterDataFormat::LocalMax0ET % wordsize_,   ClusterDataMap_[ClusterDataFormat::LocalMax0ET]);
    set32bitPart_(DataToBeEncoded.at(ClusterDataFormat::LocalMax0RelY/wordsize_), HGCC_->ClusterLocalMax0RelY(), ClusterDataFormat::LocalMax0RelY % wordsize_, ClusterDataMap_[ClusterDataFormat::LocalMax0RelY]);
    set32bitPart_(DataToBeEncoded.at(ClusterDataFormat::LocalMax0RelX/wordsize_), HGCC_->ClusterLocalMax0RelX(), ClusterDataFormat::LocalMax0RelX % wordsize_, ClusterDataMap_[ClusterDataFormat::LocalMax0RelX]);
  }
  return true; 
}


bool HGCalHardwareInterface::EncodeMulticlusterData(ClusterDataContainer& DataToBeEncoded)
{
  /* HGCalMultiCluster dataformat encoding */ 
  set32bitPart_(DataToBeEncoded.at(MultiClusterDataFormat::ET/wordsize_),             
		HGCMC_->MultiClusterET(),             
		MultiClusterDataFormat::ET % wordsize_,             
		MultiClusterDataMap_[MultiClusterDataFormat::ET]);
  set32bitPart_(DataToBeEncoded.at(MultiClusterDataFormat::Z/wordsize_),              
		HGCMC_->MultiClusterZ(),              
		MultiClusterDataFormat::Z % wordsize_,              
		MultiClusterDataMap_[MultiClusterDataFormat::Z]);
  set32bitPart_(DataToBeEncoded.at(MultiClusterDataFormat::Phi/wordsize_),            
		HGCMC_->MultiClusterPhi(),            
		MultiClusterDataFormat::Phi % wordsize_,            
		MultiClusterDataMap_[MultiClusterDataFormat::Phi]);
  set32bitPart_(DataToBeEncoded.at(MultiClusterDataFormat::Eta/wordsize_),            
		HGCMC_->MultiClusterEta(),            
		MultiClusterDataFormat::Eta % wordsize_,            
		MultiClusterDataMap_[MultiClusterDataFormat::Eta]);
  set32bitPart_(DataToBeEncoded.at(MultiClusterDataFormat::Flags/wordsize_),          
		HGCMC_->MultiClusterFlags(),          
		MultiClusterDataFormat::Flags % wordsize_,          
		MultiClusterDataMap_[MultiClusterDataFormat::Flags]);
  set32bitPart_(DataToBeEncoded.at(MultiClusterDataFormat::QFlags/wordsize_),         
		HGCMC_->MultiClusterQFlags(),         
		MultiClusterDataFormat::QFlags % wordsize_,         
		MultiClusterDataMap_[MultiClusterDataFormat::QFlags]);
  set32bitPart_(DataToBeEncoded.at(MultiClusterDataFormat::NoTriggerCells/wordsize_), 
		HGCMC_->MultiClusterNoTriggerCells(), 
		MultiClusterDataFormat::NoTriggerCells % wordsize_, 
		MultiClusterDataMap_[MultiClusterDataFormat::NoTriggerCells]);
  set32bitPart_(DataToBeEncoded.at(MultiClusterDataFormat::TBA/wordsize_),            
		HGCMC_->MultiClusterTBA(),            
		MultiClusterDataFormat::TBA % wordsize_,            
		MultiClusterDataMap_[MultiClusterDataFormat::TBA]);  
  set32bitPart_(DataToBeEncoded.at(MultiClusterDataFormat::MaxEnergyLayer/wordsize_), 
		HGCMC_->MultiClusterMaxEnergyLayer(), 
		MultiClusterDataFormat::MaxEnergyLayer % wordsize_, 
		MultiClusterDataMap_[MultiClusterDataFormat::MaxEnergyLayer]);  
  return true;
}

void HGCalHardwareInterface::InitDataMap_(){
  /* Data format bit-size for each word */  
  ClusterDataMap_[ClusterDataFormat::ET]            = 8;
  ClusterDataMap_[ClusterDataFormat::CenterY]       = 12;
  ClusterDataMap_[ClusterDataFormat::CenterX]       = 12;
  ClusterDataMap_[ClusterDataFormat::Flags]         = 6;
  ClusterDataMap_[ClusterDataFormat::SizeY]         = 8;
  ClusterDataMap_[ClusterDataFormat::SizeX]         = 8;
  ClusterDataMap_[ClusterDataFormat::LocalMax]      = 2;
  ClusterDataMap_[ClusterDataFormat::NoTriggerCells]= 8;
  ClusterDataMap_[ClusterDataFormat::LocalMax0RelY] = 8;  
  ClusterDataMap_[ClusterDataFormat::LocalMax0RelX] = 8;
  ClusterDataMap_[ClusterDataFormat::LocalMax0ET]   = 8;  
  ClusterDataMap_[ClusterDataFormat::LocalMax1RelY] = 8;  
  ClusterDataMap_[ClusterDataFormat::LocalMax1RelX] = 8;
  ClusterDataMap_[ClusterDataFormat::LocalMax1ET]   = 8;  
  ClusterDataMap_[ClusterDataFormat::LocalMax2RelY] = 8;  
  ClusterDataMap_[ClusterDataFormat::LocalMax2RelX] = 8;
  ClusterDataMap_[ClusterDataFormat::LocalMax2ET]   = 8;  
  ClusterDataMap_[ClusterDataFormat::LocalMax3RelY] = 8;  
  ClusterDataMap_[ClusterDataFormat::LocalMax3RelX] = 8;
  ClusterDataMap_[ClusterDataFormat::LocalMax3ET]   = 8;  

  MultiClusterDataMap_[MultiClusterDataFormat::BHEFraction]    = 8;
  MultiClusterDataMap_[MultiClusterDataFormat::EEEFraction]    = 8;
  MultiClusterDataMap_[MultiClusterDataFormat::ET]             = 16;
  MultiClusterDataMap_[MultiClusterDataFormat::Z]              = 10;
  MultiClusterDataMap_[MultiClusterDataFormat::Phi]            = 11;
  MultiClusterDataMap_[MultiClusterDataFormat::Eta]            = 11;
  MultiClusterDataMap_[MultiClusterDataFormat::Flags]          = 12;
  MultiClusterDataMap_[MultiClusterDataFormat::QFlags]         = 12;
  MultiClusterDataMap_[MultiClusterDataFormat::NoTriggerCells] = 8;
  MultiClusterDataMap_[MultiClusterDataFormat::TBA]            = 26;
  MultiClusterDataMap_[MultiClusterDataFormat::MaxEnergyLayer] = 6;

}

short  HGCalHardwareInterface::get32bitPart_(const std::uint32_t data32bit, const unsigned short lsb, const unsigned short size )
{
  return (data32bit >> lsb) & (0xffffffff >> (32-size));
}

void  HGCalHardwareInterface::del32bitPart_(std::uint32_t & data32bit,  const unsigned short lsb, const unsigned short size )
{
  data32bit = data32bit &  (((lsb > 0) ? (0xffffffff  >> (32-lsb)) : 0x0 ) | ((lsb+size > 31)? 0x0 : 0xffffffff  << (lsb+size)));
}

void  HGCalHardwareInterface::set32bitPart_(std::uint32_t & data32bit, const std::int32_t bitvalue,  const unsigned short lsb, const unsigned short size )
{
  if( bitvalue > std::pow(2, size) - 1 || -1 * bitvalue > std::pow(2,size-1) )
    throw std::out_of_range("the value exceeds the given bit range");
  if(size > 32 || lsb > 31 )
    throw std::out_of_range("the value exceeds 32-bit");

  del32bitPart_(data32bit,lsb, size);
  data32bit = data32bit | (bitvalue << lsb);
}


