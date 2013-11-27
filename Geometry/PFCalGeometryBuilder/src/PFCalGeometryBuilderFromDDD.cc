/** Implementation of the PFCal Geometry Builder from DDD
*
* \author P. Silva - CERN
*/

#include "Geometry/PFCalGeometryBuilder/interface/PFCalGeometryBuilderFromDDD.h"
#include "Geometry/PFCalGeometry/interface/PFCalGeometry.h"
#include "Geometry/PFCalGeometry/interface/PFCalNumberingScheme.h"

#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"

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
  PFCalNumberingScheme pfcNumberScheme;
  PFCalGeometry* geometry = new PFCalGeometry();


  DDExpandedView eview(*cview);
  std::map<DDExpandedView::nav_type,int> idMap;

  do {
    const DDLogicalPart &logPart=eview.logicalPart();
    std::string name=logPart.name();
    if(name.find("pfcal:")==std::string::npos) continue;  //this can probably be pre-filtered

    bool isPosZ( name.find("ZP") != std::string::npos);
    unsigned int lastToken( name.find_last_of("_") );
    std::string volNbStr( name.substr(lastToken+1) );
    unsigned int volNb(atoi(volNbStr.c_str()));

    //GEM drift regions
    bool isHE(false);
    int volId(volNb);
    if(volNb>=121 && volNb<=150)      { isHE=true;  volId-=121; } //1st GEM gaseous part
    else if(volNb>=181 && volNb<=210) { isHE=true;  volId-=181; } //2nd GEM gaseous part
    else if(volNb>=271 && volNb<=300) { isHE=false; volId-=271; } //Si
    else continue;

    DetId rawId=pfcNumberScheme.buildDetIdFor(isPosZ, isHE, volId, 0);

    //detector surface
    DDTranslation tran = eview.translation();
    Surface::PositionType pos(tran.x()/cm, tran.y()/cm, tran.z()/cm);

    //rotation
    DD3Vector x, y, z;
    eview.rotation().GetComponents(x,y,z);
    Surface::RotationType rot (float(x.X()), float(x.Y()), float(x.Z()),
                               float(y.X()), float(y.Y()), float(y.Z()),
                               float(z.X()), float(z.Y()), float(z.Z())); 

    //solid parameters
    std::vector<double> solidPars=eview.logicalPart().solid().parameters();
    TrapezoidalPlaneBounds *bounds = new TrapezoidalPlaneBounds(solidPars[4]/cm,solidPars[8]/cm,solidPars[0]/cm,0.4/cm);

    //the boundary plane
    BoundPlane* bp = new BoundPlane(pos, rot, bounds);
    ReferenceCountingPointer<BoundPlane> surf(bp);

    //cell specicifcation wrapper
    std::string spName = eview.logicalPart().name().name();
    PFCalCellSpecs *specs=new PFCalCellSpecs(spName,isHE ? GeomDetEnumerators::PFCalHE : GeomDetEnumerators::PFCalEE, solidPars);

    //add to the geometry
    geometry->add( new PFCalCell(rawId, surf, specs) );

  } while (eview.next() );
  
  geometry->print();

  return geometry;
}
