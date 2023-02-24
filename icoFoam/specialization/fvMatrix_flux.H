#include "fvCFD.H"


#define Type scalar
template<>
// template<class Type>
Foam::tmp<Foam::Field<Type>>
Foam::lduMatrix::faceH(const Field<Type>& psi) const
{
    if (lowerPtr_ || upperPtr_)
    {
        const scalarField& Lower = const_cast<const lduMatrix&>(*this).lower();
        const scalarField& Upper = const_cast<const lduMatrix&>(*this).upper();

        const labelUList& l = lduAddr().lowerAddr();
        const labelUList& u = lduAddr().upperAddr();

        tmp<Field<Type>> tfaceHpsi(new Field<Type> (Lower.size()));
        Field<Type> & faceHpsi = tfaceHpsi.ref();

        for (label face=0; face<l.size(); face++)
        {
            faceHpsi[face] =
                Upper[face]*psi[u[face]]
              - Lower[face]*psi[l[face]];
        }

        return tfaceHpsi;
    }
    else
    {
        FatalErrorInFunction
            << "Cannot calculate faceH"
               " the matrix does not have any off-diagonal coefficients."
            << exit(FatalError);

        return tmp<Field<Type>>(nullptr);
    }
}
#undef Type


#define Type scalar
template<>
// template<class Type>
Foam::tmp<Foam::Field<Type>>
Foam::lduMatrix::faceH(const tmp<Field<Type>>& tpsi) const
{
    tmp<Field<Type>> tfaceHpsi(faceH(tpsi()));
    tpsi.clear();
    return tfaceHpsi;
}
#undef Type

#define Type scalar
template<>
// template<class Type>
Foam::tmp<Foam::SurfaceField<Type>>
Foam::fvMatrix<Type>::
flux() const
{
    if (!psi_.mesh().schemes().fluxRequired(psi_.name()))
    {
        FatalErrorInFunction
            << "flux requested but " << psi_.name()
            << " not specified in the fluxRequired sub-dictionary"
               " of fvSchemes."
            << abort(FatalError);
    }

    // construct SurfaceField<Type>
    tmp<SurfaceField<Type>> tfieldFlux
    (
        SurfaceField<Type>::New
        (
            "flux(" + psi_.name() + ')',
            psi_.mesh(),
            dimensions()
        )
    );
    SurfaceField<Type>& fieldFlux =
        tfieldFlux.ref();

    for (direction cmpt=0; cmpt<pTraits<Type>::nComponents; cmpt++)
    {
        fieldFlux.primitiveFieldRef().replace
        (
            cmpt,
            lduMatrix::faceH(psi_.primitiveField().component(cmpt))
        );
    }

    FieldField<Field, Type> InternalContrib = internalCoeffs_;

    forAll(InternalContrib, patchi)
    {
        InternalContrib[patchi] =
            cmptMultiply
            (
                InternalContrib[patchi],
                psi_.boundaryField()[patchi].patchInternalField()
            );
    }

    FieldField<Field, Type> NeighbourContrib = boundaryCoeffs_;

    forAll(NeighbourContrib, patchi)
    {
        if (psi_.boundaryField()[patchi].coupled())
        {
            NeighbourContrib[patchi] =
                cmptMultiply
                (
                    NeighbourContrib[patchi],
                    psi_.boundaryField()[patchi].patchNeighbourField()
                );
        }
    }

    typename SurfaceField<Type>::
        Boundary& ffbf = fieldFlux.boundaryFieldRef();

    forAll(ffbf, patchi)
    {
        ffbf[patchi] = InternalContrib[patchi] - NeighbourContrib[patchi];
    }

    if (faceFluxCorrectionPtr_)
    {
        fieldFlux += *faceFluxCorrectionPtr_;
        CHECK("", "Eqn_faceFluxCorrectionPtr", *faceFluxCorrectionPtr_);
    }

    return tfieldFlux;
}

#undef Type