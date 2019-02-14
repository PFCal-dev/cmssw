#include "L1Trigger/L1THGCal/interface/concentrator/HGCalConcentratorSuperTriggerCellImpl.h"

#include <unordered_map>

HGCalConcentratorSuperTriggerCellImpl::
HGCalConcentratorSuperTriggerCellImpl(const edm::ParameterSet& conf)
  : stcSize_(conf.getParameter< std::vector<unsigned> >("stcSize")){}


int
HGCalConcentratorSuperTriggerCellImpl::getSuperTriggerCellId(int detid) const {
  HGCalDetId TC_id(detid);
  if(TC_id.subdetId()==HGCHEB) {
    return TC_id.cell(); //scintillator
  } else {

    int TC_wafer = TC_id.wafer();
    int TC_12th = ( TC_id.cell() & kSplit12_ );
    int TC_3rd = ( TC_id.cell() & kSplit3_ );

    int thickness = triggerTools_.thicknessIndex(detid,true);
    int TC_split = TC_12th;
    if (stcSize_.at(thickness) == kSTCsize16_) TC_split = TC_3rd;

    return TC_wafer<<kWafer_offset_ | TC_split;
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
