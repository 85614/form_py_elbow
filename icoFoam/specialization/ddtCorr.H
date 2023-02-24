#include "fvCFD.H"
#include "EulerDdtScheme.H"

namespace Foam {
namespace fv {


#define Type vector
template<>
// template<class Type>
tmp<typename EulerDdtScheme<Type>::fluxFieldType>
EulerDdtScheme<Type>::fvcDdtPhiCorr
(
    const VolField<Type>& U,
    const fluxFieldType& phi
)
{
    const dimensionedScalar rDeltaT = 1.0/mesh().time().deltaT();

    fluxFieldType phiCorr
    (
        phi.oldTime() - fvc::dotInterpolate(mesh().Sf(), U.oldTime())
    );

    // CHECK("", "phi", phi);
    CHECK("", "phi_old", phi.oldTime());
    CHECK("", "U_old", U.oldTime());
    CHECK("","phiCorr", phiCorr);
    return fluxFieldType::New
    (
        "ddtCorr(" + U.name() + ',' + phi.name() + ')',
        this->fvcDdtPhiCoeff(U.oldTime(), phi.oldTime(), phiCorr)
       *rDeltaT*phiCorr
    );
}
#undef Type
}} // namespace Foam::fv
