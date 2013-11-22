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
  DDFilteredView fview(*cview);
  return this->buildGeometry(fview);
}

//
PFCalGeometry* PFCalGeometryBuilderFromDDD::buildGeometry(DDFilteredView& fview)
{
  PFCalGeometry* geometry = new PFCalGeometry();
  LogDebug("PFCalGeometryBuilderFromDDD") << "I'll do something in the near future...";
  return geometry;
}
