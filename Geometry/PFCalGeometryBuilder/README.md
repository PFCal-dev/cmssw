## Setup environment

export SCRAM_ARCH=slc5_amd64_gcc472
cmsenv

## Visualize the geometry

Edit python cmsSLHCGeometryPFCalOnlyXML_cfi.py to include a new DDD 

cmsRun test/dumpPFCalGeometry_cfg.py

cmsShow --sim-geom-file ./cmsSimGeom-2.root -c test/geometry_fireworks.fwc


## Run simulation jobs

runTheMatrix.py -l 2.0 --command "--geometry Extended2023 --conditions auto:upgradePLS3"


cmsDriver.py SingleElectronPt35_cfi  --conditions auto:startup -s GEN,SIM --datatier GEN-SIM -n 10 --relval 9000,100 --eventcontent RAWSIM --geometry Extended2023 --conditions auto:upgradePLS3 --fileout file:step1.root  > step1_ProdTTbar+ProdTTbar+DIGIPROD1+RECOPROD1.log  2>&1