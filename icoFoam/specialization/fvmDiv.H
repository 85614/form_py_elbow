#include "fvCFD.H"
#include "gaussConvectionScheme.H"
#include "LimitedScheme.H"
#include "Limited01.H"
#include "limitedLinear.H"
#include <typeinfo>
#include <assert.h>


    // typedef LimitedScheme<vector, limitedLinearLimiter<NVDVTVDV>, limitFuncs::null> LimitedSchemevectorlimitedLinearLimiterNVDVTVDVnull_;
    // template <>
    // const ::Foam::word LimitedSchemevectorlimitedLinearLimiterNVDVTVDVnull_::typeName("limitedLinearV");
    // template <>
    // int LimitedSchemevectorlimitedLinearLimiterNVDVTVDVnull_::debug(::Foam::debug::debugSwitch("limitedLinearV", 0));

#define LimiterFunc NVDVTVDV
template<>
    scalar Foam::limitedLinearLimiter<LimiterFunc>::limiter
    (
        const scalar cdWeight,
        const scalar faceFlux,
        const typename LimiterFunc::phiType& phiP,
        const typename LimiterFunc::phiType& phiN,
        const typename LimiterFunc::gradPhiType& gradcP,
        const typename LimiterFunc::gradPhiType& gradcN,
        const vector& d
    ) const
    {
        scalar r = LimiterFunc::r
        (
            faceFlux, phiP, phiN, gradcP, gradcN, d
        );

        return max(min(twoByk_*r, 1), 0);
    }
#undef LimiterFunc 


#define Type Foam::vector
#define Limiter limitedLinearLimiter<NVDVTVDV>
#define LimitFunc limitFuncs::null
template<>
// template<class Type, class Limiter, template<class> class LimitFunc>
void Foam::LimitedScheme<Type, Limiter, LimitFunc>::calcLimiter
(
    const VolField<Type>& phi,
    surfaceScalarField& limiterField
) const
{
    const fvMesh& mesh = this->mesh();

    tmp<VolField<typename Limiter::phiType>> tlPhi = LimitFunc<Type>()(phi);
    const VolField<typename Limiter::phiType>& lPhi = tlPhi();

    tmp<VolField<typename Limiter::gradPhiType>> tgradc(fvc::grad(lPhi));
    const VolField<typename Limiter::gradPhiType>& gradc = tgradc();

    const surfaceScalarField& CDweights = mesh.surfaceInterpolation::weights();

    const labelUList& owner = mesh.owner();
    const labelUList& neighbour = mesh.neighbour();

    const vectorField& C = mesh.C();

    scalarField& pLim = limiterField.primitiveFieldRef();

    forAll(pLim, face)
    {
        label own = owner[face];
        label nei = neighbour[face];

        pLim[face] = Limiter::limiter
        (
            CDweights[face],
            this->faceFlux_[face],
            lPhi[own],
            lPhi[nei],
            gradc[own],
            gradc[nei],
            C[nei] - C[own]
        );
    }

    const typename VolField<Type>::Boundary&
        bPhi = phi.boundaryField();

    surfaceScalarField::Boundary& bLim =
        limiterField.boundaryFieldRef();

    forAll(bLim, patchi)
    {
        scalarField& pLim = bLim[patchi];

        if (bPhi[patchi].coupled())
        {
            const scalarField& pCDweights = CDweights.boundaryField()[patchi];
            const scalarField& pFaceFlux =
                this->faceFlux_.boundaryField()[patchi];

            const Field<typename Limiter::phiType> plPhiP
            (
                lPhi.boundaryField()[patchi].patchInternalField()
            );
            const Field<typename Limiter::phiType> plPhiN
            (
                lPhi.boundaryField()[patchi].patchNeighbourField()
            );
            const Field<typename Limiter::gradPhiType> pGradcP
            (
                gradc.boundaryField()[patchi].patchInternalField()
            );
            const Field<typename Limiter::gradPhiType> pGradcN
            (
                gradc.boundaryField()[patchi].patchNeighbourField()
            );

            // Build the d-vectors
            vectorField pd(CDweights.boundaryField()[patchi].patch().delta());

            forAll(pLim, face)
            {
                pLim[face] = Limiter::limiter
                (
                    pCDweights[face],
                    pFaceFlux[face],
                    plPhiP[face],
                    plPhiN[face],
                    pGradcP[face],
                    pGradcN[face],
                    pd[face]
                );
            }
        }
        else
        {
            pLim = 1.0;
        }
    }
}
#undef Type
#undef Limiter
#undef LimitFunc


#define Type Foam::vector
template<>
// template<class Type>
Foam::tmp<Foam::surfaceScalarField>
Foam::limitedSurfaceInterpolationScheme<Type>::weights
(
    const VolField<Type>& phi,
    const surfaceScalarField& CDweights,
    tmp<surfaceScalarField> tLimiter
) const
{
    // Note that here the weights field is initialised as the limiter
    // from which the weight is calculated using the limiter value
    surfaceScalarField& Weights = tLimiter.ref();

    scalarField& pWeights = Weights.primitiveFieldRef();

    forAll(pWeights, face)
    {
        pWeights[face] =
            pWeights[face]*CDweights[face]
          + (1.0 - pWeights[face])*pos0(faceFlux_[face]);
    }

    surfaceScalarField::Boundary& bWeights =
        Weights.boundaryFieldRef();

    forAll(bWeights, patchi)
    {
        scalarField& pWeights = bWeights[patchi];

        const scalarField& pCDweights = CDweights.boundaryField()[patchi];
        const scalarField& pFaceFlux = faceFlux_.boundaryField()[patchi];

        forAll(pWeights, face)
        {
            pWeights[face] =
                pWeights[face]*pCDweights[face]
              + (1.0 - pWeights[face])*pos0(pFaceFlux[face]);
        }
    }

    return tLimiter;
}
#undef Type


#define Type Foam::vector
template<>
// template<class Type>
Foam::tmp<Foam::surfaceScalarField>
Foam::limitedSurfaceInterpolationScheme<Type>::weights
(
    const VolField<Type>& phi
) const
{
    return this->weights
    (
        phi,
        this->mesh().surfaceInterpolation::weights(),
        this->limiter(phi)
    );
}
#undef Type


namespace Foam {
namespace fv {


    // typedef LimitedScheme<vector, limitedLinearLimiter<NVDVTVDV>, limitFuncs::null> LimitedSchemevectorlimitedLinearLimiterNVDVTVDVnull_;
    // template <>
    // const ::Foam::word LimitedSchemevectorlimitedLinearLimiterNVDVTVDVnull_::typeName("limitedLinearV");
    // template <>
    // int LimitedSchemevectorlimitedLinearLimiterNVDVTVDVnull_::debug(::Foam::debug::debugSwitch("limitedLinearV", 0));


#define Type Foam::vector
template<>
// template<class Type>
tmp<fvMatrix<Type>>
gaussConvectionScheme<Type>::fvmDiv
(
    const surfaceScalarField& faceFlux,
    const VolField<Type>& vf
) const
{
    tmp<surfaceScalarField> tweights = tinterpScheme_().weights(vf);
    typedef LimitedScheme<vector, limitedLinearLimiter<NVDVTVDV>, limitFuncs::null> LimitedSchemevectorlimitedLinearLimiterNVDVTVDVnull_;
    if (typeid(tinterpScheme_()) == typeid(LimitedSchemevectorlimitedLinearLimiterNVDVTVDVnull_))
        tweights = dynamic_cast<const LimitedSchemevectorlimitedLinearLimiterNVDVTVDVnull_ &>(tinterpScheme_()).weights(vf);
    const surfaceScalarField& weights = tweights();

    tmp<fvMatrix<Type>> tfvm
    (
        new fvMatrix<Type>
        (
            vf,
            faceFlux.dimensions()*vf.dimensions()
        )
    );
    fvMatrix<Type>& fvm = tfvm.ref();

    fvm.lower() = -weights.primitiveField()*faceFlux.primitiveField();
    fvm.upper() = fvm.lower() + faceFlux.primitiveField();
    fvm.negSumDiag();

    forAll(vf.boundaryField(), patchi)
    {
        const fvPatchField<Type>& psf = vf.boundaryField()[patchi];
        const fvsPatchScalarField& patchFlux = faceFlux.boundaryField()[patchi];
        const fvsPatchScalarField& pw = weights.boundaryField()[patchi];

        fvm.internalCoeffs()[patchi] = patchFlux*psf.valueInternalCoeffs(pw);
        fvm.boundaryCoeffs()[patchi] = -patchFlux*psf.valueBoundaryCoeffs(pw);
    }

    if (tinterpScheme_().corrected())
    {
        fvm += fvc::surfaceIntegrate(faceFlux*tinterpScheme_().correction(vf));
    }

    return tfvm;
}
#undef Type
}} // namespace Foam::fv
