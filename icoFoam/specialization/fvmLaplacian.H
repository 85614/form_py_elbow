#include "fvCFD.H"
#include "pisoControl.H"
#include "EulerDdtScheme.H"
#include "gaussConvectionScheme.H"
#include "gaussLaplacianScheme.H"

// #include "changeIO.H"

#define Type vector
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
#undef vector

#define Type vector
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
    // CHECK("", "interpolate_U", tinterpScheme_().interpolate(vsf).ref());
    // CHECK("", "grad_U", tgGrad.ref());

    correctBoundaryConditions(vsf, gGrad);

    PRINT(tgGrad.ref().boundaryField());

    return tgGrad;
}
#undef Type

#define Type vector
template<>
// template<class Type>
Foam::tmp<Foam::SurfaceField<Type>>
Foam::fv::correctedSnGrad<Type>::fullGradCorrection
(
    const VolField<Type>& vf
) const
{
    const fvMesh& mesh = this->mesh();

    // construct SurfaceField<Type>
    tmp<SurfaceField<Type>> tssf =
        linear<typename outerProduct<vector, Type>::type>(mesh).dotInterpolate
        ( 
            mesh.nonOrthCorrectionVectors(),
            gradScheme<Type>::New
            (
                mesh,
                mesh.schemes().grad("grad(" + vf.name() + ')')
            )().grad(vf, "grad(" + vf.name() + ')')
        );
    auto grad_tensor = gradScheme<Type>::New
    (
        mesh,
        mesh.schemes().grad("grad(" + vf.name() + ')')
    )().grad(vf, "grad(" + vf.name() + ')');
    // CHECK("", "U", vf);
    // CHECK("", "grad_U", grad_tensor.ref());
    tssf.ref().rename("snGradCorr(" + vf.name() + ')');
    // CHECK("", "grad_U_f", tssf.ref());

    return tssf;
}
#undef Type


#define Type vector
template<>                                                                     \
Foam::tmp<Foam::fvMatrix<Foam::Type>>                                          \
Foam::fv::gaussLaplacianScheme<Foam::Type, Foam::scalar>::fvmLaplacian         \
(                                                                              \
    const SurfaceField<scalar>& gamma,                                         \
    const VolField<Type>& vf                                                   \
)                                                                              \
{                                                                              \
    const fvMesh& mesh = this->mesh();                                         \
                                                                               \
    SurfaceField<scalar> gammaMagSf                                            \
    (                                                                          \
        gamma*mesh.magSf()                                                     \
    );                                                                         \
                                                                               \
    tmp<fvMatrix<Type>> tfvm = fvmLaplacianUncorrected                         \
    (                                                                          \
        gammaMagSf,                                                            \
        this->tsnGradScheme_().deltaCoeffs(vf),                                \
        vf                                                                     \
    );
    fvMatrix<Type>& fvm = tfvm.ref();                                          \
                                                                               \
    if (this->tsnGradScheme_().corrected())                                    \
    {              PRINT(mesh.schemes().fluxRequired(vf.name()));                                                            \
        if (mesh.schemes().fluxRequired(vf.name()))                            \
        {                                                                      \
            fvm.faceFluxCorrectionPtr() = new                                  \
            SurfaceField<Type>                   \
            (                                                                  \
                gammaMagSf*this->tsnGradScheme_().correction(vf)               \
            );                                                                 \
                                                                               \
            fvm.source() -=                                                    \
                mesh.V()*                                                      \
                fvc::div                                                       \
                (                                                              \
                    *fvm.faceFluxCorrectionPtr()                               \
                )().primitiveField();                                          \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            fvm.source() -=                                                    \
                mesh.V()*                                                      \
                fvc::div                                                       \
                (                                                              \
                    gammaMagSf*this->tsnGradScheme_().correction(vf)           \
                )().primitiveField();                                          \
        }      
        
        auto div_correction =  mesh.V()*                                                      \
                fvc::div                                                       \
                (                                                              \
                    gammaMagSf*this->tsnGradScheme_().correction(vf)           \
                )().primitiveField();
        // CHECK("", "grad_U_f", this->tsnGradScheme_().correction(vf).ref());
        // CHECK("", "div_correction", div_correction.ref());
        // CHECK("", "laplacian_nu_U[SOURCE]", fvm.source());
    }                                                                          \
                                                                               \
    return tfvm;                                                               \
}                                                                              \


#undef Type
