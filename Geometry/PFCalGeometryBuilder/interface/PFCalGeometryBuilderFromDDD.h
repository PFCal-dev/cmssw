#ifndef PFCalGeometry_PFCalGeometryBuilderFromDDD_H
#define PFCalGeometry_PFCalGeometryBuilderFromDDD_H

/** \class PFCalGeometryBuilderFromDDD
 *
* Build the PFCalGeometry ftom the DDD description
*
* \author P. Silva - CERN
*
*/

#include <string>
#include <map>
#include <list>

class DDCompactView;
class DDFilteredView;
class PFCalGeometry;

class PFCalGeometryBuilderFromDDD
{
 public:

  /**
     @short CTOR
  */
  PFCalGeometryBuilderFromDDD();

  /**
     @short DTOR
  */
  ~PFCalGeometryBuilderFromDDD();

  /**
     @short instantiate new geometry from xml
  */
  PFCalGeometry* build(const DDCompactView* cview);


 private:

  PFCalGeometry* buildGeometry(const DDCompactView *fview);

};

#endif
