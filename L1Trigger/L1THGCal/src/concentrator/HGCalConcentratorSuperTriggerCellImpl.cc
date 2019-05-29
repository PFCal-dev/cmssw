#include "L1Trigger/L1THGCal/interface/concentrator/HGCalConcentratorSuperTriggerCellImpl.h"

HGCalConcentratorSuperTriggerCellImpl::
HGCalConcentratorSuperTriggerCellImpl(const edm::ParameterSet& conf)
  : fixedDataSizePerHGCROC_(conf.getParameter<bool>("fixedDataSizePerHGCROC")),
    coarseTCmapping_(std::vector<unsigned>{HGCalCoarseTriggerCellMapping::kCTCsizeVeryFine_,HGCalCoarseTriggerCellMapping::kCTCsizeVeryFine_,HGCalCoarseTriggerCellMapping::kCTCsizeVeryFine_,HGCalCoarseTriggerCellMapping::kCTCsizeVeryFine_}),
    superTCmapping_(conf.getParameter<std::vector<unsigned>>("stcSize"))
{

    std::string energyType(conf.getParameter<string>("type_energy_division"));

    if( energyType == "superTriggerCell" ){
      energyDivisionType_ = superTriggerCell;
    }else if( energyType == "oneBitFraction" ){
      energyDivisionType_ = oneBitFraction;

      oneBitFractionThreshold_ = conf.getParameter<double>("oneBitFractionThreshold");
      oneBitFractionLowValue_ = conf.getParameter<double>("oneBitFractionLowValue");
      oneBitFractionHighValue_ = conf.getParameter<double>("oneBitFractionHighValue");

    }else if( energyType == "equalShare" ){
      energyDivisionType_ = equalShare;

    }else {
      energyDivisionType_ = superTriggerCell;
    } 


    
}

void 
HGCalConcentratorSuperTriggerCellImpl::
createAllTriggerCells( std::unordered_map<unsigned,SuperTriggerCell>& STCs, std::vector<l1t::HGCalTriggerCell>& trigCellVecOutput) const
{ 

    for (auto& s: STCs){

      int thickness = 0;
      if ( triggerTools_.isSilicon(s.second.getSTCId() ) ){
        thickness = triggerTools_.thicknessIndex(s.second.getSTCId(),true);
      }else if ( triggerTools_.isScintillator(s.second.getSTCId()) ){
        thickness = 3;
      }

      std::vector<uint32_t> output_ids = superTCmapping_.getConstituentTriggerCells( s.second.getSTCId() );

      for (const auto& id: output_ids){

        if ( fixedDataSizePerHGCROC_ && thickness > kHighDensityThickness_ && id!=superTCmapping_.getRepresentativeDetId(id)) {
          continue;
        }

        if ( !triggerTools_.validTriggerCell(id) ) {
          continue;
        }
        
        l1t::HGCalTriggerCell triggerCell;

        GlobalPoint point;
        if ( fixedDataSizePerHGCROC_ == true && thickness > kHighDensityThickness_ ){
          point = coarseTCmapping_.getCoarseTriggerCellPosition( coarseTCmapping_.getCoarseTriggerCellId(id) );
        }
        else{
          point = triggerTools_.getTCPosition(id);
        }
        math::PtEtaPhiMLorentzVector p4(0, point.eta(), point.phi(), 0.);
        triggerCell.setPosition(point);
        triggerCell.setP4(p4);
        triggerCell.setDetId(id);
	if ( energyDivisionType_ == superTriggerCell && id != s.second.getMaxId() ){
	  continue;
	}
        trigCellVecOutput.push_back ( triggerCell );

        if ( energyDivisionType_ == oneBitFraction ){ //Get the 1 bit fractions

          if ( id != s.second.getMaxId() ){
            float tc_fraction = getTriggerCellOneBitFraction( s.second.getTCpt(id) , s.second.getSumPt() );
            s.second.addToFractionSum(tc_fraction);
          }
        }

      }
    }

    // assign energy
    for ( l1t::HGCalTriggerCell & tc : trigCellVecOutput){
      const auto & stc = STCs[superTCmapping_.getCoarseTriggerCellId(tc.detId())]; 
      assignSuperTriggerCellEnergy(tc, stc);        
    }      

}

void
HGCalConcentratorSuperTriggerCellImpl::
 assignSuperTriggerCellEnergy(l1t::HGCalTriggerCell &c, const SuperTriggerCell &stc) const {
      
  if ( energyDivisionType_ == superTriggerCell ){
    if ( c.detId() == stc.getMaxId() ){
      c.setHwPt(stc.getSumHwPt());
      c.setMipPt(stc.getSumMipPt());
      c.setPt(stc.getSumPt());
    }
    else {
      c.setHwPt(0);
      c.setMipPt(0);
      c.setPt(0);
    }
  }
  else if ( energyDivisionType_ == equalShare ){

    double denominator =  fixedDataSizePerHGCROC_ ? double(kTriggerCellsForDivision_) : double(stc.size());

    c.setHwPt( stc.getSumHwPt() / denominator );
    c.setMipPt( stc.getSumMipPt() / denominator );
    c.setPt( stc.getSumPt() /denominator );
  }
  else if (  energyDivisionType_ == oneBitFraction ){
    
    double frac = 0;
    
    if ( c.detId() != stc.getMaxId() ){
      frac = getTriggerCellOneBitFraction( c.pt(), stc.getSumPt() );
    }
    else{
      frac = 1-stc.getFractionSum();
    }
    
    c.setHwPt( stc.getSumHwPt() * frac );
    c.setMipPt( stc.getSumMipPt() * frac );
    c.setPt( stc.getSumPt() * frac );
  }
  
}

float 
HGCalConcentratorSuperTriggerCellImpl::
getTriggerCellOneBitFraction( float tcPt, float sumPt ) const{

  double f = tcPt / sumPt ;
  double frac = 0;
  if ( f < oneBitFractionThreshold_ ){
    frac = oneBitFractionLowValue_;
  }
  else{
    frac = oneBitFractionHighValue_;
  }

  return frac;

}

void 
HGCalConcentratorSuperTriggerCellImpl::
select(const std::vector<l1t::HGCalTriggerCell>& trigCellVecInput, std::vector<l1t::HGCalTriggerCell>& trigCellVecOutput)
{ 

  std::unordered_map<unsigned,SuperTriggerCell> STCs;
  // first pass, fill the "coarse" trigger cells
  for (const l1t::HGCalTriggerCell & tc : trigCellVecInput) {

    uint32_t stcid = superTCmapping_.getCoarseTriggerCellId(tc.detId());
    STCs[stcid].add(tc, stcid);

  }

  createAllTriggerCells( STCs, trigCellVecOutput);

}
