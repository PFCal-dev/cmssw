=== Visualize the geometry

Edit python cmsSLHCGeometryPFCalOnlyXML_cfi.py to include a new DDD 

cmsRun test/dumpPFCalGeometry_cfg.py

cmsShow --sim-geom-file ./cmsSimGeom-2.root -c test/geometry_fireworks.fwc
