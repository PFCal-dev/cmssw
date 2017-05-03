#ifndef __L1Trigger_L1THGCal_HGCalHardwareInterface_h__
#define __L1Trigger_L1THGCal_HGCalHardwareInterface_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class HGCalHardwareInterface{
  
 public:
  /* mapping is retrieved from: https://twiki.cern.ch/twiki/bin/viewauth/CMS/HGCALTriggerArchitectureDesignAndImplementation */
  struct ClusterDataFormat{
    enum CDF {ET = 0, CenterY = 8, CenterX = 20, Flags = 32, SizeY = 38, SizeX = 46, LocalMax = 54, NoTriggerCells = 56, LocalMax1RelX = 64,  LocalMax0ET = 72, LocalMax0RelY = 80, LocalMax0RelX = 88, LocalMax2RelY = 96,  LocalMax2RelX = 104,  LocalMax1ET = 112,  LocalMax1RelY = 120,  LocalMax3ET = 128, LocalMax3RelY = 136,  LocalMax3RelX = 144,  LocalMax2ET = 152 };
  };
  struct MultiClusterDataFormat{
    enum MCDF {BHEFraction = 0, EEEFraction = 8, ET = 16, Z = 32, Phi = 42, Eta = 53, Flags = 64, QFlags = 76, NoTriggerCells = 88, TBA = 96, MaxEnergyLayer = 122};
  };
  
  typedef std::vector<uint32_t> ClusterDataContainer;
  typedef std::map<int,int>     DataFormatMap;

  HGCalHardwareInterface(l1t::HGCalCluster * HGCC){
    this->HGCC_  = HGCC;
  };

  HGCalHardwareInterface(l1t::HGCalMulticluster * HGCMC){
    this->HGCMC_ = HGCMC;

  };

  bool DecodeMulticlusterData (const ClusterDataContainer& DataToBeDecoded);
	
  bool EncodeMulticlusterData (ClusterDataContainer& DataToBeEncoded);

  bool DecodeClusterData (const ClusterDataContainer& DataToBeDecoded);
	
  bool EncodeClusterData (ClusterDataContainer& DataToBeEncoded);

  
 private:
        
  HGCalHardwareInterface(){};
  
  l1t::HGCalMulticluster * HGCMC_;
  l1t::HGCalCluster      * HGCC_;

  /* Object required to have a low-latency data encode / decode scheme */
  ClusterDataContainer MultiClusterData_;
  
  DataFormatMap MultiClusterDataMap_;
  
  DataFormatMap ClusterDataMap_;

  /*the data format relies on 32-bit word */  
  static const int wordsize_ = 32;
  
  void InitDataMap_();
  
  /* returns the part of the 32 bit word (which is defined by the LSB and size). Signed values are required to be converted in the decoder class.  */
  short get32bitPart_(const std::uint32_t data32bit, const unsigned short lsb,    const unsigned short size);
  
  /* deletes part of the 32-bit word, required for the set function */
  void  del32bitPart_(std::uint32_t & data32bit,     const unsigned short lsb,    const unsigned short size);
  
  /* sets part of the 32-bit data word */ 
  void  set32bitPart_(std::uint32_t & data32bit,     const std::int32_t bitvalue, const unsigned short lsb, const unsigned short size );
  
};

#endif
