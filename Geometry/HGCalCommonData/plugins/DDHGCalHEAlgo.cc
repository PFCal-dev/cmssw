///////////////////////////////////////////////////////////////////////////////
// File: DDHGCalHEAlgo.cc
// Description: Geometry factory class for HGCal (EE)
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/HGCalCommonData/plugins/DDHGCalHEAlgo.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDHGCalHEAlgo::DDHGCalHEAlgo() {
  edm::LogInfo("HGCalGeom") << "DDHGCalHEAlgo info: Creating an instance";
}

DDHGCalHEAlgo::~DDHGCalHEAlgo() {}

void DDHGCalHEAlgo::initialize(const DDNumericArguments & nArgs,
			       const DDVectorArguments & vArgs,
			       const DDMapArguments & ,
			       const DDStringArguments & sArgs,
			       const DDStringVectorArguments &vsArgs){

  materials     = vsArgs["MaterialNames"];
  names         = vsArgs["VolumeNames"];
  thick         = vArgs["ThicknessType"];
  type          = dbl_to_int(vArgs["LayerType"]);
  zMinBlock     = vArgs["ZMinType"];
  offsets       = dbl_to_int(vArgs["Offsets"]);
  rotstr        = sArgs["Rotation"];
  layers        = (int)(nArgs["Layers"]);
  thickModule   = nArgs["ThickModule"];
  edm::LogInfo("HGCalGeom") << "DDHGCalHEAlgo: " << materials.size()
			    << " volumes to be put with rotation " << rotstr
			    << " in " << layers << " layers with " 
			    << thickModule << " gaps";
  for (unsigned int i=0; i<names.size(); ++i)
    edm::LogInfo("HGCalGeom") << "Volume [" << i << "] " << names[i]
			      << " filled with " << materials[i] << " of type "
			      << type[i] << " thickness " << thick[i]
			      << " starting at " << zMinBlock[i] << " with"
			      << " copy number " << offsets[i];

  slopeB        = nArgs["SlopeBottom"];
  slopeT        = vArgs["SlopeTop"];
  zFront        = vArgs["ZFront"];
  rMaxFront     = vArgs["RMaxFront"];
  sectors       = (int)(nArgs["Sectors"]);
  idName        = parent().name().name();
  idNameSpace   = DDCurrentNamespace::ns();
  edm::LogInfo("HGCalGeom") << "DDHGCalHEAlgo: Bottom slope " << slopeB
			    << " " << slopeT.size() << " slopes for top";
  for (unsigned int i=0; i<slopeT.size(); ++i)
    edm::LogInfo("HGCalGeom") << "Block [" << i << "] Zmin " << zFront[i]
			      << " Rmax " << rMaxFront[i] << " Slope " 
			      << slopeT[i];
  edm::LogInfo("HGCalGeom") << "DDHGCalHEAlgo: Sectors " << sectors
			    << "\tNameSpace:Name " << idNameSpace
			    << ":" << idName;

}

////////////////////////////////////////////////////////////////////
// DDHGCalHEAlgo methods...
////////////////////////////////////////////////////////////////////

void DDHGCalHEAlgo::execute(DDCompactView& cpv) {
  
  edm::LogInfo("HGCalGeom") << "==>> Constructing DDHGCalHEAlgo...";
  constructLayers (parent(), cpv);
  edm::LogInfo("HGCalGeom") << "<<== End of DDHGCalHEAlgo construction ...";
}

void DDHGCalHEAlgo::constructLayers(DDLogicalPart module, DDCompactView& cpv) {
  
  edm::LogInfo("HGCalGeom") << "DDHGCalHEAlgo test: \t\tInside Layers";

  ///////////////////////////////////////////////////////////////
  //Pointers to the Rotation Matrices and to the Materials
  DDRotation rot(DDName(DDSplit(rotstr).first, DDSplit(rotstr).second));

  for (unsigned int ii=0; ii<materials.size(); ii++) {
    DDName matName(DDSplit(materials[ii]).first, DDSplit(materials[ii]).second);
    DDMaterial matter(matName);
    double  zi    = zMinBlock[ii];
    int     copy  = offsets[ii];
    int     ityp  = type[ii];
    for (int i=0; i<layers; i++) {
      double  zo     = zi + thick[ii];
      double  rinF   = zi * slopeB;
      double  rinB   = zo * slopeB;
      double  routF  = rMax(zi);
      double  routB  = rMax(zo);
      std::string name = "HGCal"+names[ii]+dbl_to_string(copy);
      edm::LogInfo("HGCalGeom") << "DDHGCalHEAlgo test: Layer " << i << ":" 
				<< ii << " Front " << zi << ", " << rinF 
				<< ", " << routF << " Back " << zo 
				<< ", " << rinB << ", " << routB;
      DDHGCalHEAlgo::HGCalHEPar parm = (ityp == 0) ?
	parameterLayer(rinF, routF, rinB, routB, zi, zo) :
	parameterLayer(ityp, rinF, routF, rinB, routB, zi, zo);
      DDSolid solid = DDSolidFactory::trap(DDName(name, idNameSpace), 
					   0.5*thick[ii], parm.theta,
					   parm.phi, parm.yh1, parm.bl1, 
					   parm.tl1, parm.alp, parm.yh2,
					   parm.bl2, parm.tl2, parm.alp);

      DDLogicalPart glog = DDLogicalPart(solid.ddname(), matter, solid);
      edm::LogInfo("HGCalGeom") << "DDHGCalHEAlgo test: " 
				<< solid.name() << " Trap made of " << matName
				<< " of dimensions " << 0.5*thick[ii] << ", "
				<< parm.theta/CLHEP::deg << ", " 
				<< parm.phi/CLHEP::deg << ", " << parm.yh1 
				<< ", " << parm.bl1 << ", " << parm.tl1 
				<< ", " << parm.alp/CLHEP::deg << ", " 
				<< parm.yh2 << ", " << parm.bl2 << ", " 
				<< parm.tl2 << ", " << parm.alp/CLHEP::deg;
      DDTranslation r1(parm.xpos, parm.ypos, parm.zpos);
      cpv.position(glog, module, copy, r1, rot);
      edm::LogInfo("HGCalGeom") << "DDHGCalHEAlgo test: " << glog.name()
				<< " number " << copy << " positioned in " 
				<< module.name() << " at " << r1 << " with " 
				<< rot;
      zi  += thickModule;
      copy++;
      ityp = -ityp;
    }   // End of loop on layers
  }     // End of loop on types
}


DDHGCalHEAlgo::HGCalHEPar 
DDHGCalHEAlgo::parameterLayer(double rinF, double routF, double rinB, 
			      double routB, double zi, double zo) {

  DDHGCalHEAlgo::HGCalHEPar parm;
  //Given rin, rout compute parameters of the trapezoid and 
  //position of the trapezoid for a standrd layer
  double alpha = CLHEP::pi/sectors;
  edm::LogInfo("HCalGeom") << "Input: Front " << rinF << " " << routF << " "
			   << zi << " Back " << rinB << " " << routB << " "
			   << zo << " Alpha " << alpha/CLHEP::deg;

  parm.yh2  = parm.yh1  = 0.5 * (routF*cos(alpha) - rinB);
  parm.bl2  = parm.bl1  = rinB  * tan(alpha);
  parm.tl2  = parm.tl1  = routF * sin(alpha);
  parm.xpos = 0.5*(routF*cos(alpha)+rinB);
  parm.ypos = 0.0;
  parm.zpos = 0.5*(zi+zo);
  parm.alp  = parm.theta  = parm.phi = 0;
  edm::LogInfo("HCalGeom") << "Output Dimensions " << parm.yh1 << " " 
			   << parm.bl1 << " " << parm.tl1 << " " << parm.yh2 
			   << " " << parm.bl2 << " " << parm.tl2 << " " 
			   << parm.alp/CLHEP::deg <<" " <<parm.theta/CLHEP::deg
			   << " " << parm.phi/CLHEP::deg << " Position " 
			   << parm.xpos << " " << parm.ypos << " " <<parm.zpos;
  return parm;
}

DDHGCalHEAlgo::HGCalHEPar 
DDHGCalHEAlgo::parameterLayer(int type, double rinF, double routF, double rinB,
			      double routB, double zi, double zo) {

  DDHGCalHEAlgo::HGCalHEPar parm;
  //Given rin, rout compute parameters of the trapezoid and 
  //position of the trapezoid for a standrd layer
  double alpha = CLHEP::pi/sectors;
  edm::LogInfo("HGCalGeom") << "Input " << type << " Front " << rinF << " " 
			    << routF << " " << zi << " Back " << rinB << " " 
			    << routB << " " << zo <<" Alpha " 
			    << alpha/CLHEP::deg;

  parm.yh2  = parm.yh1  = 0.5 * (routF*cos(alpha) - rinB);
  parm.bl2  = parm.bl1  = 0.5 * rinB  * tan(alpha);
  parm.tl2  = parm.tl1  = 0.5 * routF * sin(alpha);
  double dx = 0.25* (parm.bl2+parm.tl2-parm.bl1-parm.tl1);
  double dy = 0.0;
  parm.xpos = 0.5*(routF*cos(alpha)+rinB);
  parm.ypos = 0.25*(parm.bl2+parm.tl2+parm.bl1+parm.tl1);
  parm.zpos = 0.5*(zi+zo);
  parm.alp  = atan(0.5 * tan(alpha));
  if (type > 0) {
    parm.ypos = -parm.ypos;
  } else {
    parm.alp  = -parm.alp;
    dx        = -dx;
  }
  double r    = sqrt (dx*dx + dy*dy);
  edm::LogInfo("HGCalGeom") << "dx|dy|r " << dx << ":" << dy << ":" << r;
  if (r > 1.0e-8) {
    parm.theta  = atan (r/(zo-zi));
    parm.phi    = atan2 (dy, dx);
  } else {
    parm.theta  = parm.phi = 0;
  }
  edm::LogInfo("HGCalGeom") << "Output Dimensions " << parm.yh1 << " " 
			    << parm.bl1 << " " << parm.tl1 << " " << parm.yh2 
			    << " " << parm.bl2 << " " << parm.tl2 << " " 
			    << parm.alp/CLHEP::deg <<" " <<parm.theta/CLHEP::deg
			    << " " << parm.phi/CLHEP::deg << " Position " 
			    << parm.xpos << " " << parm.ypos << " " <<parm.zpos;
  return parm;
}

double DDHGCalHEAlgo::rMax(double z) {

  double r(0);
  for (unsigned int k=0; k<slopeT.size(); ++k) {
    if (z < zFront[k]) break;
    r = rMaxFront[k] + (z - zFront[k]) * slopeT[k];
  }
  return r;
}
