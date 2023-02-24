#include "fvCFD.H"

#define Type scalar
template<>
// template<class Type>
Foam::tmp<Foam::Field<Type>> Foam::fvPatchField<Type>::snGrad() const
{
    return patch_.deltaCoeffs()*(*this - patchInternalField());
}
#undef Type

namespace Foam {
namespace fv { // 这两个可以不要的，像上面这个Foam::fvPatchField<Type>::snGrad()，但是，有些符号是在Foam::fv里的，又不想用using namespace


#define Type scalar
template<>
// template<class Type>
Foam::tmp
<
    Foam::VolField<typename Foam::outerProduct<Foam::vector, Type>::type>
>
Foam::fv::gaussGrad<Type>::gradf
(
    const SurfaceField<Type>& ssf,
    const word& name
)
{
    typedef typename outerProduct<vector, Type>::type GradType;

    const fvMesh& mesh = ssf.mesh();

    tmp<VolField<GradType>> tgGrad
    (
        VolField<GradType>::New
        (
            name,
            mesh,
            dimensioned<GradType>
            (
                "0",
                ssf.dimensions()/dimLength,
                Zero
            ),
            extrapolatedCalculatedFvPatchField<GradType>::typeName
        )
    );
    VolField<GradType>& gGrad = tgGrad.ref();

    const labelUList& owner = mesh.owner();
    const labelUList& neighbour = mesh.neighbour();
    const vectorField& Sf = mesh.Sf();

    Field<GradType>& igGrad = gGrad;
    const Field<Type>& issf = ssf;

    forAll(owner, facei)
    {
        GradType Sfssf = Sf[facei]*issf[facei];

        igGrad[owner[facei]] += Sfssf;
        igGrad[neighbour[facei]] -= Sfssf;
    }

    forAll(mesh.boundary(), patchi)
    {
        const labelUList& pFaceCells =
            mesh.boundary()[patchi].faceCells();

        const vectorField& pSf = mesh.Sf().boundaryField()[patchi];

        const fvsPatchField<Type>& pssf = ssf.boundaryField()[patchi];

        forAll(mesh.boundary()[patchi], facei)
        {
            igGrad[pFaceCells[facei]] += pSf[facei]*pssf[facei];
        }
    }

    igGrad /= mesh.V();

    gGrad.correctBoundaryConditions();

    return tgGrad;
}
#undef Type


#define Type scalar
template<>
// template<class Type>
void Foam::fv::gaussGrad<Type>::correctBoundaryConditions
(
    const VolField<Type>& vsf,
    VolField<typename outerProduct<vector, Type>::type>& gGrad
);

template<>
// template<class Type>
Foam::tmp
<
    Foam::VolField<typename Foam::outerProduct<Foam::vector, Type>::type>
>
Foam::fv::gaussGrad<Type>::calcGrad
(
    const VolField<Type>& vsf,
    const word& name
) const
{
    typedef typename outerProduct<vector, Type>::type GradType;

    tmp<VolField<GradType>> tgGrad
    (
        gradf(tinterpScheme_().interpolate(vsf), name)
    );
    VolField<GradType>& gGrad = tgGrad.ref();

    correctBoundaryConditions(vsf, gGrad);

    return tgGrad;
}
#undef Type


#define Type scalar
template<>
// template<class Type>
void Foam::fv::gaussGrad<Type>::correctBoundaryConditions
(
    const VolField<Type>& vsf,
    VolField<typename outerProduct<vector, Type>::type>& gGrad
)
{
    typename VolField<typename outerProduct<vector, Type>::type>::Boundary&
        gGradbf = gGrad.boundaryFieldRef();

    forAll(vsf.boundaryField(), patchi)
    {
        if (!vsf.boundaryField()[patchi].coupled())
        {
            const vectorField n
            (
                vsf.mesh().Sf().boundaryField()[patchi]
              / vsf.mesh().magSf().boundaryField()[patchi]
            );

            gGradbf[patchi] += n *
            (
                vsf.boundaryField()[patchi].snGrad()
              - (n & gGradbf[patchi])
            );
        }
     }
}


#undef Type
}} // namespace Foam::fv