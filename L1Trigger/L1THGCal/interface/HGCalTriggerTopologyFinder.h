#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "DataFormats/ForwardDetId/interface/HGCTriggerDetId.h"

#include "Geometry/FCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <map>

/* Original Author: Andrea Carlo Marini
 * email : andrea.carlo.marini@cern.ch
 * date : 09/10/2015
 *
 * This plugins want to read the HGCal geometry/topology and fill the Trigger geometry with a navigation system.
 * May be slow, but generic
 */

#ifndef HGCAL_TRIGGERTOPOLOGY_FINDER_H
#define HGCAL_TRIGGERTOPOLOGY_FINDER_H 1

namespace HGCalTriggerGeometry{


    // base class for a geometry modifier
    class HGCalTriggerGeometryModifier{
        // * general class to be derived from for a geometry modifier, passed into the init
        protected:
            const HGCalGeometry & mGeometry;

            // --- the trigger cell modifier base class is friend of TriggerCell and Module: access to their private members 
            void insertNeighbour(TriggerCell* tc, unsigned tc_id, HGCalTriggerGeometry::TopoBits tb)
            {
                tc->neighbours_ . insert( std::make_pair(tc_id,tb));
            }
            void insertNeighbour(Module* mod, unsigned mod_id, HGCalTriggerGeometry::TopoBits tb)
            {
                mod -> neighbours_ . insert ( std::make_pair(mod_id,tb));
            }
            HGCalTriggerGeometryBase::module_construction_container & get_modules_inconstruction( HGCalTriggerGeometryBase &b) { return b.modules_inconstruction_ ;}
            HGCalTriggerGeometryBase::trigger_cell_construction_container & get_triggercells_inconstruction( HGCalTriggerGeometryBase &b) { return b.triggercells_inconstruction_ ;}

            // -- getTrigger ... In Construction
            HGCalTriggerGeometry::TriggerCell* getTriggerCellFromCellInConstruction( HGCalTriggerGeometryBase &b, unsigned id) { return b . getTriggerCellFromCellInConstruction(id )  ;}
            HGCalTriggerGeometry::Module* getModuleFromCellInConstruction(HGCalTriggerGeometryBase &b, unsigned id){ return b.getModuleFromCellInConstruction( id ) ;}
            HGCalTriggerGeometry::Module* getModuleFromTriggerCellInConstruction(HGCalTriggerGeometryBase &b, unsigned id){ return b.getModuleFromTriggerCellInConstruction( id ) ;}

        public:
            HGCalTriggerGeometryModifier( const HGCalGeometry &g): mGeometry(g) {} 
            virtual int initialize(HGCalTriggerGeometryBase &g )  =0;

    };


    // ----------------------- FACTORY ------------------------
    class Creator;

    // this is a singleton
    class HGCalTriggerGeometryModifierFactory{
        private: 
            //Singleton
            HGCalTriggerGeometryModifierFactory(){};
            HGCalTriggerGeometryModifierFactory(HGCalTriggerGeometryModifierFactory const&)  = delete;
            void operator=(HGCalTriggerGeometryModifierFactory const&)  = delete;
            //
            std::map<std::string, HGCalTriggerGeometry::Creator*> table_;
        public:
            // singleton
            static HGCalTriggerGeometryModifierFactory& get(){ static HGCalTriggerGeometryModifierFactory instance; return instance; }
            //
            HGCalTriggerGeometryModifier* create(const std::string &name,  const HGCalGeometry &g) ;
            void inline registerit(const std::string& classname, Creator* creator){
                std::cout<<"[HGCalTriggerGeometryModifierFactory]"<<"::[INFO]"<<"Registering "<<classname<<std::endl;
                table_[classname] = creator;
            };
    };

    class Creator{
        public:
            virtual HGCalTriggerGeometryModifier* create(const HGCalGeometry &g)=0;
            Creator(const std::string& classname) { std::cout<<" called creator const for"<<classname<<std::endl; HGCalTriggerGeometryModifierFactory::get() . registerit(classname,this) ; }
    };
    // creator
    template <class T>
        class CreatorImpl : public HGCalTriggerGeometry::Creator
    {
        public:
            CreatorImpl(const std::string&classname) : Creator(classname){}
            HGCalTriggerGeometryModifier* create( const HGCalGeometry &g) override final{ return new T(g); }
    };

#define REGISTER(classname)\
    namespace { \
    HGCalTriggerGeometry::CreatorImpl<classname> _creator_ (#classname); \
    };

    // ------------------------------------------------------
    class HGCalTriggerTopologyFinder : public HGCalTriggerGeometry::HGCalTriggerGeometryModifier {	
        // const HGCalTopology & mTopology; geometry knows about topology


        // fill tr cell neigh
        int initTriggerCells(HGCalTriggerGeometryBase &);
        //fill tr module neigh
        int initTriggerModules(HGCalTriggerGeometryBase &);


        public:

        HGCalTriggerTopologyFinder( const HGCalGeometry &g): HGCalTriggerGeometryModifier(g) {} 

        // this is const 
        inline int initialize(HGCalTriggerGeometryBase &g ) 
        { return initTriggerCells(g) | initTriggerModules(g) ; }

    };


}; // namespace

#endif

// Local Variables:
// mode:c++
// indent-tabs-mode:nil
// tab-width:4
// c-basic-offset:4
// End:
// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 
