#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTopologyFinder.h"

using namespace HGCalTriggerGeometry;

namespace {
  std::unique_ptr<const TriggerCell> null_tc;
  std::unique_ptr<const Module>      null_mod;
}

HGCalTriggerGeometryBase::
HGCalTriggerGeometryBase(const edm::ParameterSet& conf) : 
  name_(conf.getParameter<std::string>("TriggerGeometryName")),
  ee_sd_name_(conf.getParameter<std::string>("eeSDName")),
  fh_sd_name_(conf.getParameter<std::string>("fhSDName")),
  bh_sd_name_(conf.getParameter<std::string>("bhSDName")) {
}

void HGCalTriggerGeometryBase::reset() {
  geom_map().swap(cells_to_trigger_cells_);
  geom_map().swap(trigger_cells_to_modules_);
  module_map().swap(modules_);
  trigger_cell_map().swap(trigger_cells_);
}

const std::unique_ptr<const TriggerCell>& 
HGCalTriggerGeometryBase::
getTriggerCellFromCell( const unsigned cell_det_id ) const {
  auto found_tc = cells_to_trigger_cells_.find(cell_det_id);
  if( found_tc == cells_to_trigger_cells_.end() ) {
    return null_tc;
  }
  return trigger_cells_.find(found_tc->second)->second;
}

const std::unique_ptr<const Module>& 
HGCalTriggerGeometryBase::
getModuleFromCell( const unsigned cell_det_id ) const {
  auto found_tc = cells_to_trigger_cells_.find(cell_det_id);
  if( found_tc == cells_to_trigger_cells_.end() ) {
    return null_mod;
  }
  auto found_mod = trigger_cells_to_modules_.find(found_tc->second);
  if( found_mod == trigger_cells_to_modules_.end() ) {
    return null_mod;
  }
  return modules_.find(found_mod->second)->second;
}

const std::unique_ptr<const Module>&
HGCalTriggerGeometryBase::
getModuleFromTriggerCell( const unsigned trigger_cell_det_id ) const {
  auto found_mod = trigger_cells_to_modules_.find(trigger_cell_det_id);
  if( found_mod == trigger_cells_to_modules_.end() ) {
    return null_mod;
  }
  return modules_.find(found_mod->second)->second;
}

void HGCalTriggerGeometryBase::initModifiers(){ 
	 for (auto &gMod : geometryModifiers_) gMod -> initialize( *this); 
 }

// the unique pointers don't loose ownership
HGCalTriggerGeometry::TriggerCell* HGCalTriggerGeometryBase::getTriggerCellFromCellInConstruction( const unsigned cell_det_id) { 
        auto found_tc = cells_to_trigger_cells_.find(cell_det_id) ; 
        if (found_tc == cells_to_trigger_cells_.end()) return (TriggerCell*)(NULL);
        return triggercells_inconstruction_.find( found_tc ->second)->second.get(); 
    }

HGCalTriggerGeometry::Module* HGCalTriggerGeometryBase::getModuleFromCellInConstruction( const unsigned cell_det_id) { 
        auto found_tc = cells_to_trigger_cells_.find(cell_det_id) ; 
        if (found_tc == cells_to_trigger_cells_.end() )
	{
		return (Module*)(NULL); 
	}
	return getModuleFromTriggerCellInConstruction(found_tc->second);

}

HGCalTriggerGeometry::Module* HGCalTriggerGeometryBase::getModuleFromTriggerCellInConstruction( const unsigned trcell_det_id) {    
        auto found_mod = trigger_cells_to_modules_.find( trcell_det_id);
        if(found_mod == trigger_cells_to_modules_.end() ) return (Module*)(NULL);
        return modules_inconstruction_ . find( found_mod->second) ->second.get();
    }


void HGCalTriggerGeometryBase::finalizeInitialization(){ 
        // --- make maps const. destroy non const maps
        // these pointeres are released from the inconstruction and cost cast to be placed in the final geometry
	
	// check if modules_ and trigger_cells_ are empty
	if (not modules_ . empty() )
	{
		edm::LogWarning("HGCalTriggerGeometryBase") <<
			" modules map is not empty in finalization of the geometry modifiers. ";
	}
	if (not trigger_cells_ . empty() )
	{
		edm::LogWarning("HGCalTriggerGeometryBase") <<
			" trigger cells map is not empty in finalization of the geometry modifiers";
	}
	//---      
        for( auto& p : modules_inconstruction_) 
            modules_ . insert( std::make_pair(p.first, std::unique_ptr<const HGCalTriggerGeometry::Module >(p.second.release()))) ;
        for(auto&p : triggercells_inconstruction_ ) 
            trigger_cells_ . insert( std::make_pair(p.first, std::unique_ptr<const HGCalTriggerGeometry::TriggerCell>(p.second.release())) );
        modules_inconstruction_.clear();
        triggercells_inconstruction_.clear();
 } 


EDM_REGISTER_PLUGINFACTORY(HGCalTriggerGeometryFactory,
			   "HGCalTriggerGeometryFactory");
