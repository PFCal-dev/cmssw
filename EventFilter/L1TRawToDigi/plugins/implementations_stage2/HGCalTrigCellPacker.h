#ifndef L1T_PACKER_STAGE2_HGCALTRIGCELLPACKER_H
#define L1T_PACKER_STAGE2_HGCALTRIGCELLPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

namespace l1t {
   namespace stage2 {
      class HGCalTrigCellPacker : public Packer {
         public:
            Blocks pack(const edm::Event&, const PackerTokens*) override;
      };
   }
}

#endif
