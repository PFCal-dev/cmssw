/** Implementation of the PFCal Geometry Builder from DDD
*
* \author P. Silva - CERN
*/

#include "Geometry/PFCalGeometryBuilder/interface/PFCalGeometryBuilderFromDDD.h"
#include "Geometry/PFCalGeometry/interface/PFCalGeometry.h"

#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"

#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"

#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <iostream>
#include <algorithm>

//
PFCalGeometryBuilderFromDDD::PFCalGeometryBuilderFromDDD() {}

//
PFCalGeometryBuilderFromDDD::~PFCalGeometryBuilderFromDDD() {}

//
PFCalGeometry* PFCalGeometryBuilderFromDDD::build(const DDCompactView* cview)
{
  return this->buildGeometry( cview );
}

using namespace std;
//
PFCalGeometry* PFCalGeometryBuilderFromDDD::buildGeometry(const DDCompactView *cview)
{
  PFCalGeometry* geometry = new PFCalGeometry();

  DDExpandedView eview(*cview);
  std::map<DDExpandedView::nav_type,int> idMap;

  do {
    const DDLogicalPart &logPart=eview.logicalPart();
    std::string name=logPart.name();
    if(name.find("pfcal:")==std::string::npos) continue;  //this can probably be pre-filtered

    bool posZ( name.find("ZP") != std::string::npos);
    unsigned int lastToken( name.find_last_of("_") );
    std::string volNbStr( name.substr(lastToken+1) );
    unsigned int volNb(atoi(volNbStr.c_str()));

    //GEM drift regions
    if((volNb>=121 && volNb<=150) || (volNb>=181 && volNb<=210))
      {
	
      }
    //Si wafers
    else if(volNb>=271 && volNb<=300)
      {
      }
    else continue;
    
    //position
    DD3Vector x, y, z;
    eview.rotation().GetComponents(x,y,z);
    cout << name << " " << posZ 
	 << " (" << eview.translation().x() << "," << eview.translation().y() << "," << eview.translation().z() << ") "
	 << " (" << x.X() << "," << y.X() << "," << z.X() << ") " 
	 << " (" << x.Y() << "," << y.Y() << "," << z.Y() << ") " 
	 << " (" << x.Z() << "," << y.Z() << "," << z.Z() << ") " << endl;


  } while (eview.next() );
  
    
  LogDebug("PFCalGeometryBuilderFromDDD") << "I'll do something in the near future...";
  return geometry;
}
