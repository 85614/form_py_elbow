#include "fvCFD.H"

/////////////////////////////// UEqn.A() begin ////////////////////////////
#define Type Foam::vector
#define PatchField Foam::fvPatchField
#define GeoMesh Foam::volMesh
template<>
// template<class Type, template<class> class PatchField, class GeoMesh>
void Foam::GeometricBoundaryField<Type, PatchField, GeoMesh>::evaluate()
{
    if (GeometricField<Type, PatchField, GeoMesh>::debug)
    {
        InfoInFunction << endl;
    }

    if
    (
        Pstream::defaultCommsType == Pstream::commsTypes::blocking
     || Pstream::defaultCommsType == Pstream::commsTypes::nonBlocking
    )
    {
        label nReq = Pstream::nRequests();

        forAll(*this, patchi)
        {
            this->operator[](patchi).initEvaluate(Pstream::defaultCommsType);
        }

        // Block for any outstanding requests
        if
        (
            Pstream::parRun()
         && Pstream::defaultCommsType == Pstream::commsTypes::nonBlocking
        )
        {
            Pstream::waitRequests(nReq);
        }

        forAll(*this, patchi)
        {
            this->operator[](patchi).evaluate(Pstream::defaultCommsType);
        }
    }
    else if (Pstream::defaultCommsType == Pstream::commsTypes::scheduled)
    {
        const lduSchedule& patchSchedule =
            bmesh_.mesh().globalData().patchSchedule();

        forAll(patchSchedule, patchEvali)
        {
            if (patchSchedule[patchEvali].init)
            {
                this->operator[](patchSchedule[patchEvali].patch)
                    .initEvaluate(Pstream::commsTypes::scheduled);
            }
            else
            {
                this->operator[](patchSchedule[patchEvali].patch)
                    .evaluate(Pstream::commsTypes::scheduled);
            }
        }
    }
    else
    {
        FatalErrorInFunction
            << "Unsupported communications type "
            << Pstream::commsTypeNames[Pstream::defaultCommsType]
            << exit(FatalError);
    }
}
#undef Type
#undef PatchField
#undef GeoMesh


#define Type Foam::vector
#define PatchField Foam::fvPatchField
#define GeoMesh Foam::volMesh
template<>
// template<class Type, template<class> class PatchField, class GeoMesh>
void Foam::GeometricField<Type, PatchField, GeoMesh>::
correctBoundaryConditions()
{
    this->setUpToDate();
    storeOldTimes();
    boundaryField_.evaluate();
}
#undef Type
#undef PatchField
#undef GeoMesh

#define Type vector
template<>
// template<class Type>
Foam::tmp<Foam::volScalarField> Foam::fvMatrix<Type>::A() const
{
    tmp<volScalarField> tAphi
    (
        volScalarField::New
        (
            "A(" + psi_.name() + ')',
            psi_.mesh(),
            dimensions_/psi_.dimensions()/dimVol,
            extrapolatedCalculatedFvPatchScalarField::typeName
        )
    );

    tAphi.ref().primitiveFieldRef() = D()/psi_.mesh().V();
    tAphi.ref().correctBoundaryConditions();

    return tAphi;
}
#undef Type

/////////////////////////////// UEqn.A() end ////////////////////////////