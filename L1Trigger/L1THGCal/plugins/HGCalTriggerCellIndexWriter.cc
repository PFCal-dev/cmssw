#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "TTree.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HGCTriggerDetId.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include <stdlib.h> 


class HGCalTriggerCellIndexWriter : public edm::EDAnalyzer 
{
    public:
        explicit HGCalTriggerCellIndexWriter(const edm::ParameterSet& );
        ~HGCalTriggerCellIndexWriter();

        virtual void beginRun(const edm::Run&, const edm::EventSetup&);
        virtual void analyze(const edm::Event&, const edm::EventSetup&);


    private:
        void writeTriggerCellMapping(const HGCalTriggerGeometryBase::es_info&);
        std::unique_ptr<HGCalTriggerGeometryBase> triggerGeometry_; 
        //
        int sector_;
        int zside_;
        int layer_;
        int module_;
        //
        std::string outputFile_;
};


/*****************************************************************/
HGCalTriggerCellIndexWriter::HGCalTriggerCellIndexWriter(const edm::ParameterSet& conf):
    sector_(conf.getParameter<int>("Sector")),
    zside_(conf.getParameter<int>("ZSide")),
    layer_(conf.getParameter<int>("Layer")),
    module_(conf.getParameter<int>("Module")),
    outputFile_(conf.getParameter<std::string>("OutputFile"))
/*****************************************************************/
{
    //setup geometry 
    const edm::ParameterSet& geometryConfig = conf.getParameterSet("TriggerGeometry");
    const std::string& trigGeomName = geometryConfig.getParameter<std::string>("TriggerGeometryName");
    HGCalTriggerGeometryBase* geometry = HGCalTriggerGeometryFactory::get()->create(trigGeomName,geometryConfig);
    triggerGeometry_.reset(geometry);

}



/*****************************************************************/
HGCalTriggerCellIndexWriter::~HGCalTriggerCellIndexWriter() 
/*****************************************************************/
{
}

/*****************************************************************/
void HGCalTriggerCellIndexWriter::beginRun(const edm::Run& /*run*/, 
                                          const edm::EventSetup& es)
/*****************************************************************/
{
    triggerGeometry_->reset();
    HGCalTriggerGeometryBase::es_info info;
    const std::string& ee_sd_name = triggerGeometry_->eeSDName();
    const std::string& fh_sd_name = triggerGeometry_->fhSDName();
    const std::string& bh_sd_name = triggerGeometry_->bhSDName();
    es.get<IdealGeometryRecord>().get(ee_sd_name,info.geom_ee);
    es.get<IdealGeometryRecord>().get(fh_sd_name,info.geom_fh);
    es.get<IdealGeometryRecord>().get(bh_sd_name,info.geom_bh);
    es.get<IdealGeometryRecord>().get(ee_sd_name,info.topo_ee);
    es.get<IdealGeometryRecord>().get(fh_sd_name,info.topo_fh);
    es.get<IdealGeometryRecord>().get(bh_sd_name,info.topo_bh);
    triggerGeometry_->initialize(info);

    writeTriggerCellMapping(info);
}


/*****************************************************************/
void HGCalTriggerCellIndexWriter::writeTriggerCellMapping(const HGCalTriggerGeometryBase::es_info& info)
/*****************************************************************/
{
    std::map<uint32_t,uint32_t> cellsIndex;
    std::map<uint32_t,uint32_t> cellsSortedIndex;
    std::multimap<uint32_t,uint32_t> triggerCellsAndCells;

    // Loop over modules and choose the specified module
    for( const auto& id_module : triggerGeometry_->modules() )
    {

        HGCTriggerDetId id(id_module.first);
        const auto& modulePtr = id_module.second;
        // Use only the chosen module
        if(id.zside()!=zside_ ||
                id.layer()!=layer_ ||
                id.sector()!=sector_ ||
                id.module()!=module_) continue;

        // fill (trigger cell, cell) list into a sorted map
        for(const auto& tc_c : modulePtr->triggerCellComponents())
        {
            cellsIndex.insert( std::make_pair(tc_c.second, 0) );
            triggerCellsAndCells.insert( std::make_pair(tc_c.first, tc_c.second) );
        }
        // translate the cell ID into an index inside the module
        uint32_t index = 0;
        for(auto& c_i : cellsIndex) 
        {
            c_i.second = index;
            index++;
        }
        // loop over sorted (trigger cell, cell) and associate sorted index to cells.
        // "sorted" here means that cells inside a given trigger cell are grouped together
        // and ordered from trigger cell 1 to trigger cell N in the module
        index = 0;
        for(const auto& tc_c : triggerCellsAndCells)
        {
            cellsSortedIndex.insert( std::make_pair(index, tc_c.second) );
            index++;
        }

        // loop over sorted cells and print index
        std::vector<uint32_t> cellsIndexToPrint;
        for(const auto& i_c : cellsSortedIndex)
        {
            uint32_t originalIndex = cellsIndex.at(i_c.second);
            cellsIndexToPrint.push_back(originalIndex);
        }

        std::fstream output(outputFile_, std::ios::out);
        // Print in file in reversed order
        for(std::vector<uint32_t>::const_reverse_iterator itr=cellsIndexToPrint.crbegin(); itr!=cellsIndexToPrint.crend(); ++itr)
        {
            output<<*itr<<"\n";
        }
        output.close();
    }
}


/*****************************************************************/
void HGCalTriggerCellIndexWriter::analyze(const edm::Event& e, 
			      const edm::EventSetup& es) 
/*****************************************************************/
{

}



//define this as a plug-in
DEFINE_FWK_MODULE(HGCalTriggerCellIndexWriter);
