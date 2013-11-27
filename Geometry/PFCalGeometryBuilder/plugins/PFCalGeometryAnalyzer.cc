#include "Geometry/PFCalGeometryBuilder/plugins/PFCalGeometryAnalyzer.h"
#include "Geometry/PFCalGeometryBuilder/interface/PFCalGeometryBuilderFromDDD.h"

#include "DetectorDescription/Core/interface/adjgraph.h"
#include "DetectorDescription/Core/interface/graphwalker.h"
#include "DetectorDescription/Core/interface/graph_util.h"


#include "DetectorDescription/OfflineDBLoader/interface/GeometryInfoDump.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include <iostream>

using namespace std;

//
PFCalGeometryAnalyzer::PFCalGeometryAnalyzer( const edm::ParameterSet &iConfig )
{
  listRoots_           = iConfig.getUntrackedParameter<bool>("listRoots",    false);
  showSerialized_      = iConfig.getUntrackedParameter<bool>("showSerialized",    false);
  ddViewName_          = iConfig.getUntrackedParameter<std::string>("ddViewName",     "");
  ddRootNodeName_      = iConfig.getUntrackedParameter<std::string>("ddRootNodeName", "cms:OCMS");
  geomInfoDumpCfg_     = iConfig.getParameter<edm::ParameterSet>("geomInfoDumpCfg");
  runGeometryInfoDump_ = geomInfoDumpCfg_.getUntrackedParameter<bool>("run",false);
  runGeometryBuilderFromDDD_ = iConfig.getUntrackedParameter<bool>("runGeometryBuilderFromDDD",true);
}

//
PFCalGeometryAnalyzer::~PFCalGeometryAnalyzer()
{
}

//
void PFCalGeometryAnalyzer::analyze( const edm::Event &iEvent, const edm::EventSetup &iSetup)
{
  cout << "[PFCalGeometryAnalyzer::analyze] start" << endl;

  //get handle to the DD
  edm::ESTransientHandle<DDCompactView> ddViewH;
  iSetup.get<IdealGeometryRecord>().get( ddViewName_, ddViewH );

  //safety check
  if (!ddViewH.isValid() || !ddViewH.description()) {
    cout << "[PFCalGeometryAnalyzer::analyze] Handle for DD is not valid or does not contain any valid description" << endl;
    return;
  }

  const DDCompactView &pToDDView=*ddViewH;

  if(listRoots_)      listRoots(pToDDView);
  if(showSerialized_) showSerialized(pToDDView,"");
  if(runGeometryInfoDump_)
    {
      GeometryInfoDump gidump;
      gidump.dumpInfo( geomInfoDumpCfg_.getUntrackedParameter<bool>("dumpGeoHistory",false),
		       geomInfoDumpCfg_.getUntrackedParameter<bool>("dumpSpecs", false),
		       geomInfoDumpCfg_.getUntrackedParameter<bool>("dumpPosInfo", false),
		       pToDDView,
		       geomInfoDumpCfg_.getUntrackedParameter<std::string>("outFileName", "GeoHistory"),
		       geomInfoDumpCfg_.getUntrackedParameter<uint32_t>("numNodesToDump", 0));
    }
  if(runGeometryBuilderFromDDD_)
    {
      PFCalGeometryBuilderFromDDD dddBuilder;
      dddBuilder.build( &pToDDView );
    }

  cout << "[PFCalGeometryAnalyzer::analyze] end" << endl;
}

//
void PFCalGeometryAnalyzer::listRoots(const DDCompactView  &dd)
{
  cout << "[PFCalGeometryAnalyzer::listRoots]" << endl;
  const DDCompactView::graph_type &g=dd.graph();
  DDCompactView::graph_type::edge_list roots;
  g.findRoots(roots);
  while (roots.size()) {
    cout << g.nodeData(roots.back().first) << ' ';
    roots.pop_back();
  }  
  cout << endl;
}

//
void PFCalGeometryAnalyzer::showSerialized(const DDCompactView  &dd, const string & root)
{
  cout << "[PFCalGeometryAnalyzer::serialize]" << endl;
  const DDCompactView::graph_type &g=dd.graph();
  DDCompactView::walker_type w(g);//,root);
  bool go(true);
  while(go) {
    cout << w.current().first << " " << endl;
    go=w.next();
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(PFCalGeometryAnalyzer);
