#ifndef DataFormats_L1Trigger_HGCalClusterT_h
#define DataFormats_L1Trigger_HGCalClusterT_h

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/ClusterShapes.h"
#include "Math/Vector3D.h"


namespace l1t 
{
  template <class C> class HGCalClusterT : public L1Candidate 
  {

    public:
      typedef typename edm::PtrVector<C>::const_iterator const_iterator;

    public:
      HGCalClusterT(){}
      HGCalClusterT( const LorentzVector p4,
          int pt=0,
          int eta=0,
          int phi=0
          )
        : L1Candidate(p4, pt, eta, phi),
        valid_(true),
        detId_(0),
        centre_(0, 0, 0),
        centreProj_(0., 0., 0.),
        mipPt_(0),
        seedMipPt_(0){}

      HGCalClusterT( const edm::Ptr<C>& c ):
        valid_(true)
      {
        addConstituent(c);
      }
      
      ~HGCalClusterT() {};
      
      const edm::PtrVector<C>& constituents() const {return constituents_;}        
      const_iterator constituents_begin() const {return constituents_.begin();}
      const_iterator constituents_end() const {return constituents_.end();}
      unsigned size() const { return constituents_.size(); }

      void addConstituent( const edm::Ptr<C>& c )
      {
        if( constituents_.empty() )
        { 
          detId_ = HGCalDetId(c->detId());
          seedMipPt_ = c->mipPt();
        }

	
	if(clusterXmin_ > c->position().x()){
	  clusterXmin_ = c->position().x();
	} else if(clusterXmax_ < c->position().x()){
	  clusterXmax_ = c->position().x();
	}
	if(clusterYmin_ > c->position().y()){
	  clusterYmin_ = c->position().y();
	} else if(clusterYmax_ < c->position().y()){
	  clusterYmax_ = c->position().y();
	}

        /* update cluster positions */
        Basic3DVector<float> constituentCentre( c->position() );
        Basic3DVector<float> clusterCentre( centre_ );

        clusterCentre = clusterCentre*mipPt_ + constituentCentre*c->mipPt();
        if( mipPt_ + c->mipPt()!=0 ) 
        {
          clusterCentre /= ( mipPt_ + c->mipPt() ) ;
        }
        centre_ = GlobalPoint( clusterCentre );


        if( clusterCentre.z()!=0 ) 
        {
          centreProj_= GlobalPoint( clusterCentre / clusterCentre.z() );
        }
        /* update cluster energies */
        mipPt_ += c->mipPt();

        int updatedPt = hwPt() + c->hwPt();
        setHwPt(updatedPt);

        math::PtEtaPhiMLorentzVector updatedP4 ( p4() );
        updatedP4 += c->p4(); 
        setP4( updatedP4 );

        constituents_.push_back( c );

      }
      
      bool valid() const { return valid_;}
      void setValid(bool valid) { valid_ = valid;}
      
      double clusterXspread() const { return (clusterXmax_ - clusterXmin_);}
      double clusterYspread() const { return (clusterYmax_ - clusterYmin_);}
      double mipPt() const { return mipPt_; }
      double seedMipPt() const { return seedMipPt_; }
      uint32_t detId() const { return detId_.rawId(); }

      /* distance in 'cm' */
      double distance( const l1t::HGCalTriggerCell &tc ) const 
      {
        return ( tc.position() - centre_ ).mag();
      }

      const GlobalPoint& position() const { return centre_; } 
      const GlobalPoint& centre() const { return centre_; }
      const GlobalPoint& centreProj() const { return centreProj_; }

      // FIXME: will need to fix places where the shapes are directly accessed
      // Right now keep shapes() getter as non-const 
      ClusterShapes& shapes() {return shapes_;}
      double hOverE() const
      {
        double pt_em = 0.;
        double pt_had = 0.;
        double hOe = 0.;

        for(const auto& constituent : constituents())
        {
          switch( constituent->subdetId() )
          {
            case HGCEE:
              pt_em += constituent->pt();
              break;
            case HGCHEF:
              pt_had += constituent->pt();
              break;
            case HGCHEB:
              pt_had += constituent->pt();
              break;
            default:
              break;
          }
        }
        if(pt_em>0) hOe = pt_had / pt_em ;
        else hOe = -1.;
        return hOe;
      }

      uint32_t subdetId() const {return detId_.subdetId();} 
      uint32_t layer() const {return detId_.layer();}
      int32_t zside() const {return detId_.zside();}

      int32_t cluster2DET()             const {return cluster2DET_;}
      int32_t cluster2DCenterX()        const {return cluster2DCenterX_;}
      int32_t cluster2DCenterY()        const {return cluster2DCenterY_;}
      int32_t cluster2DSizeY()          const {return cluster2DSizeY_;}
      int32_t cluster2DSizeX()          const {return cluster2DSizeX_;}
      int32_t cluster2DNoTriggerCells() const {return cluster2DNoTriggerCells_;}
      int32_t cluster2DnLocalMaxima()   const {return cluster2DnLocalMaxima_;}
      int32_t cluster2DLocalMax0ET()    const {return cluster2DLocalMax0ET_;}
      int32_t cluster2DLocalMax0RelY()  const {return cluster2DLocalMax0RelY_;}
      int32_t cluster2DLocalMax0RelX()  const {return cluster2DLocalMax0RelX_;}
      int32_t cluster2DLocalMax1ET()    const {return cluster2DLocalMax1ET_;}
      int32_t cluster2DLocalMax1RelY()  const {return cluster2DLocalMax1RelY_;}
      int32_t cluster2DLocalMax1RelX()  const {return cluster2DLocalMax1RelX_;}
      int32_t cluster2DLocalMax2ET()    const {return cluster2DLocalMax2ET_;}
      int32_t cluster2DLocalMax2RelY()  const {return cluster2DLocalMax2RelY_;}
      int32_t cluster2DLocalMax2RelX()  const {return cluster2DLocalMax2RelX_;}
      int32_t cluster2DLocalMax3ET()    const {return cluster2DLocalMax3ET_;}
      int32_t cluster2DLocalMax3RelY()  const {return cluster2DLocalMax3RelY_;}
      int32_t cluster2DLocalMax3RelX()  const {return cluster2DLocalMax3RelX_;}
      
      void setCluster2DET(const int32_t CET)               { cluster2DET_ = CET;}
      void setCluster2DCenterX(const int32_t CCX)          { cluster2DCenterX_ = CCX;}
      void setCluster2DCenterY(const int32_t CCY)          { cluster2DCenterY_ = CCY;}
      void setCluster2DSizeY(const int32_t CSY)            { cluster2DSizeY_ = CSY;}
      void setCluster2DSizeX(const int32_t CSX)            { cluster2DSizeX_ = CSX;}
      void setCluster2DNoTriggerCells(const int32_t CNTC)  { cluster2DNoTriggerCells_ = CNTC;}
      void setCluster2DnLocalMaxima(const int32_t CLM)     { cluster2DnLocalMaxima_ = CLM;}
      void setCluster2DLocalMax0ET(const int32_t CLM0ET)   { cluster2DLocalMax0ET_ = CLM0ET;}
      void setCluster2DLocalMax0RelY(const int32_t CLM0RY) { cluster2DLocalMax0RelY_ = CLM0RY;}
      void setCluster2DLocalMax0RelX(const int32_t CLM0RX) { cluster2DLocalMax0RelX_ = CLM0RX;}
      void setCluster2DLocalMax1ET(const int32_t CLM1ET)   { cluster2DLocalMax1ET_ = CLM1ET;}
      void setCluster2DLocalMax1RelY(const int32_t CLM1RY) { cluster2DLocalMax1RelY_ = CLM1RY;}
      void setCluster2DLocalMax1RelX(const int32_t CLM1RX) { cluster2DLocalMax1RelX_ = CLM1RX;}
      void setCluster2DLocalMax2ET(const int32_t CLM2ET)   { cluster2DLocalMax2ET_ = CLM2ET;}
      void setCluster2DLocalMax2RelY(const int32_t CLM2RY) { cluster2DLocalMax2RelY_ = CLM2RY;}
      void setCluster2DLocalMax2RelX(const int32_t CLM2RX) { cluster2DLocalMax2RelX_ = CLM2RX;}
      void setCluster2DLocalMax3ET(const int32_t CLM3ET)   { cluster2DLocalMax3ET_ = CLM3ET;}
      void setCluster2DLocalMax3RelY(const int32_t CLM3RY) { cluster2DLocalMax3RelY_ = CLM3RY;}
      void setCluster2DLocalMax3RelX(const int32_t CLM3RX) { cluster2DLocalMax3RelX_ = CLM3RX;}
      
      int32_t cluster3DBHEFraction()    const {return cluster3DBHEFraction_;}
      int32_t cluster3DEEEFraction()    const {return cluster3DEEEFraction_;}
      int32_t cluster3DET()             const {return cluster3DET_;}
      int32_t cluster3DZ()              const {return cluster3DZ_;}
      int32_t cluster3DPhi()            const {return cluster3DPhi_;}
      int32_t cluster3DEta()            const {return cluster3DEta_;}
      int32_t cluster3DFlags()          const {return cluster3DFlags_;}
      int32_t cluster3DQFlags()         const {return cluster3DQFlags_;}
      int32_t cluster3DNoTriggerCells() const {return cluster3DNoTriggerCells_;}
      int32_t cluster3DTBA()            const {return cluster3DTBA_;}
      int32_t cluster3DMaxEnergyLayer() const {return cluster3DMaxEnergyLayer_;}
      
      void setCluster3DBHEFraction(const int32_t BHE )     { cluster3DBHEFraction_ = BHE;}
      void setCluster3DEEEFraction(const int32_t EEE)      { cluster3DEEEFraction_ = EEE;}
      void setCluster3DET(const int32_t MCET )             { cluster3DET_ = MCET;}
      void setCluster3DZ(const int32_t MCZ)                { cluster3DZ_ = MCZ;}
      void setCluster3DPhi(const int32_t MCP)              { cluster3DPhi_ = MCP;}
      void setCluster3DEta(const int32_t MCE )             { cluster3DEta_ = MCE;}
      void setCluster3DFlags(const int32_t MCF )           { cluster3DFlags_ = MCF;}
      void setCluster3DQFlags(const int32_t MCQF)          { cluster3DQFlags_ = MCQF ;}
      void setCluster3DNoTriggerCells(const int32_t MCNTC) { cluster3DNoTriggerCells_ = MCNTC;}
      void setCluster3DTBA(const int32_t MCTBA)            { cluster3DTBA_ = MCTBA;}
      void setCluster3DMaxEnergyLayer(const int32_t MCMEL) { cluster3DMaxEnergyLayer_ = MCMEL;}
      

      /* operators */
      bool operator<(const HGCalClusterT<C>& cl) const {return mipPt() < cl.mipPt();}
      bool operator>(const HGCalClusterT<C>& cl) const  { return  cl<*this;   }
      bool operator<=(const HGCalClusterT<C>& cl) const { return !(cl>*this); }
      bool operator>=(const HGCalClusterT<C>& cl) const { return !(cl<*this); }

    private:
        
      bool valid_;
      HGCalDetId detId_;     
      edm::PtrVector<C> constituents_;
      GlobalPoint centre_;
      GlobalPoint centreProj_; // centre projected onto the first HGCal layer


      int32_t cluster2DET_            = 0;
      int32_t cluster2DCenterX_       = 0;
      int32_t cluster2DCenterY_       = 0;
      int32_t cluster2DSizeY_         = 0;
      int32_t cluster2DSizeX_         = 0;
      int32_t cluster2DNoTriggerCells_= 0;
      int32_t cluster2DnLocalMaxima_  = 0;
      int32_t cluster2DLocalMax0ET_   = 0;
      int32_t cluster2DLocalMax0RelY_ = 0;
      int32_t cluster2DLocalMax0RelX_ = 0;
      int32_t cluster2DLocalMax1ET_   = 0;
      int32_t cluster2DLocalMax1RelY_ = 0;
      int32_t cluster2DLocalMax1RelX_ = 0;
      int32_t cluster2DLocalMax2ET_   = 0;
      int32_t cluster2DLocalMax2RelY_ = 0;
      int32_t cluster2DLocalMax2RelX_ = 0;
      int32_t cluster2DLocalMax3ET_   = 0;
      int32_t cluster2DLocalMax3RelY_ = 0;
      int32_t cluster2DLocalMax3RelX_ = 0;

      int32_t cluster3DBHEFraction_   = 0;
      int32_t cluster3DEEEFraction_   = 0;
      int32_t cluster3DET_            = 0;
      int32_t cluster3DZ_             = 0;
      int32_t cluster3DPhi_           = 0;
      int32_t cluster3DEta_           = 0;
      int32_t cluster3DFlags_         = 0;
      int32_t cluster3DQFlags_        = 0;
      int32_t cluster3DNoTriggerCells_= 0;
      int32_t cluster3DTBA_           = 0;
      int32_t cluster3DMaxEnergyLayer_= 0;

      double clusterYmin_ = 99999.;
      double clusterXmin_ = 99999.;
      double clusterYmax_ = 0.;
      double clusterXmax_ = 0.;

      double mipPt_;
      double seedMipPt_;

      ClusterShapes shapes_;

  };

}

#endif
