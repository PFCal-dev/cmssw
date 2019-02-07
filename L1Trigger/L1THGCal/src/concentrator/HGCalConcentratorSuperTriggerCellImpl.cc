#include "L1Trigger/L1THGCal/interface/concentrator/HGCalConcentratorSuperTriggerCellImpl.h"

#include <unordered_map>

HGCalConcentratorSuperTriggerCellImpl::
HGCalConcentratorSuperTriggerCellImpl(const edm::ParameterSet& conf){}


int
HGCalConcentratorSuperTriggerCellImpl::getSuperTriggerCellId(int detid) const {
  // FIXME: won't work in the V9 geometry
  HGCalDetId TC_id(detid);
  if(TC_id.subdetId()==HGCHEB) {
    return TC_id.cell(); //scintillator
  } else {
    int TC_wafer = TC_id.wafer();

    int split_0 = 0x3a;
    int split_2 = 0x30;
    int thickness = triggerTools_.thicknessIndex(detid);

    int TC_12th = 0;
    if ( thickness == 0 || thickness == 1)
      TC_12th = ( TC_id.cell() & split_0 );
    if ( thickness == 2)
      TC_12th = ( TC_id.cell() & split_0 );
    int wafer_offset = 6;
    return TC_wafer<<wafer_offset | TC_12th;
  }
  
}

void 
HGCalConcentratorSuperTriggerCellImpl::
superTriggerCellSelectImpl(const std::vector<l1t::HGCalTriggerCell>& trigCellVecInput, std::vector<l1t::HGCalTriggerCell>& trigCellVecOutput)
{ 

  std::unordered_map<unsigned,SuperTriggerCell> STCs;

  // first pass, fill the super trigger cells
  for (const l1t::HGCalTriggerCell & tc : trigCellVecInput) {
    if (tc.subdetId() == HGCHEB) continue;
    STCs[getSuperTriggerCellId(tc.detId())].add(tc);
  }
    
  // second pass, write them out
  for (const l1t::HGCalTriggerCell & tc : trigCellVecInput) {
    
    //If scintillator use a simple threshold cut
    if (tc.subdetId() == HGCHEB) {
      trigCellVecOutput.push_back( tc );
    } else {
      const auto & stc = STCs[getSuperTriggerCellId(tc.detId())]; 
      if (tc.detId() == stc.GetMaxId() ) {
        trigCellVecOutput.push_back( tc );
        stc.assignEnergy(trigCellVecOutput.back());
      }
    }
    
  } // end of second loop
  
}
