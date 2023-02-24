/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2021 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    icoFoam

Description
    Transient solver for incompressible, laminar flow of Newtonian fluids.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "pisoControl.H"
#include "changeIO2.H"


bool use_info = true;


#define PAIR(x) #x, (x)
void __print()
{
}
template <typename _First, typename... _Others>
void __print(_First &&first, _Others &&...others)
{
    print2py(Info, std::forward<_First>(first));
    __print(std::forward<_Others>(others)...);
}
template <typename... _Args>
void print(_Args &&...args)
{
    if (use_info)
    {
        __print("'''\n", std::forward<_Args>(args)..., "\n'''\n"); // 默认end="\n"
    }
}
#define LOG(msg, x) print("log(" #msg ", \""  #x "\",\n", (x), ")")
#define SET(msg, x) print("set(" #msg ", \""  #x "\",\n", (x), ")")
#define CHECK(msg, var_name, true_value) print("check(msg=", #msg, ", var=" #var_name ", foam_var=\"" #true_value "\", expected=\n", (true_value), ")")
constexpr auto hexfloat = ios_base::fixed | ios_base::scientific;

#define PRINT(x) (void)(Info << #x " begin\n" << (x) << "\n" #x " end" << endl)
#define PRINT_EXPR(x) (void)(Info << #x " = " << (x) << endl)

#include "specialization/fvMatrix_A.H"
#include "specialization/ddtCorr.H"
#include "specialization/grad.H"
#include "specialization/fvmLaplacian.H"
#include "specialization/fvMatrix_flux.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    Info << "'''" << endl;
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"

    pisoControl piso(mesh);

    #include "createFields.H"
    #include "initContinuityErrs.H"

    // static_cast<Foam::OSstream &>(Info).setf(hexfloat, std::ios::floatfield);
    // static_cast<Foam::OSstream &>(Info).setf(hexfloat, std::ios::floatfield);
    // std::cout << std::hexfloat;
    if (physicalProperties.lookupOrDefault<Foam::label>("hexfloat", 1))
    {
        Info.operator Foam::OSstream &().setf(ios_base::fixed | ios_base::scientific, std::ios::floatfield);
    }

    print("import sys\n" "sys.path.append('..')");
    print("from icoFoamPy.base_elbow import *");

    SET("", mesh.faceAreas());
    SET("", mesh.faceOwner());
    SET("", mesh.faceNeighbour());
    SET("", mesh.cellVolumes());
    SET("", mesh.delta().ref());
    SET("", mesh.deltaCoeffs());
    SET("", mesh.nonOrthDeltaCoeffs());
    SET("", mesh.nonOrthCorrectionVectors());
    SET("", phi.mesh().surfaceInterpolation::weights());
    SET("", phi.mesh().nonOrthDeltaCoeffs());
    
    SET("", runTime.deltaT().value());
    SET("", nu.value());

    // PRINT(U);
    // PRINT(U.boundaryField());
    // PRINT(p);
    // PRINT(p.boundaryField());
    // PRINT(phi);
    // PRINT(phi.boundaryField());
    // Info << U;
    // Info << U.boundaryField();
    {
        List<label> boundary_start;
        List<label> boundary_end;

        for (auto &b : mesh.boundary())
        {

            boundary_start.append(b.start());
            boundary_end.append(b.start() + b.size());
        }
        SET("", boundary_start);
        SET("", boundary_end);
    }

    {
        List<word> boundary_u_type;
        List<word> boundary_p_type;
        for (auto &ub : U.boundaryField())
        {
            boundary_u_type.append(ub.type());
        }
        for (auto &pb : p.boundaryField())
        {
            boundary_p_type.append(pb.type());
        }
        SET("", boundary_p_type);
        SET("", boundary_u_type);
    }
    SET("", U);
    SET("", p);
    SET("", phi);

    print("init_variable()");

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting time loop\n" << endl;

    int loop_count = 0;

    while (runTime.loop())
    {
        ++loop_count;
        use_info = loop_count <= 5;

        print("print('\\nloop ", loop_count, "')");
        Info<< "Time = " << runTime.userTimeName() << nl << endl;

        #include "CourantNo.H"

        // Momentum predictor

        print("make_UEqn()");
        
        fvVectorMatrix UEqn
        (
            fvm::ddt(U)
          + fvm::div(phi, U)
          - fvm::laplacian(nu, U)
        );

        CHECK("Eqn tmp-var", "ddt_U", fvm::ddt(U).ref());
        // CHECK("", "phi", phi);
        CHECK("Eqn tmp-var", "div_phi_U", fvm::div(phi, U).ref());
        // PRINT(fvm::laplacian(nu, U).ref());
        // PRINT(U);
        // CHECK("tmp-var", "laplacian_nu_U[UPPER]", fvm::laplacian(nu, U).ref().upper());
        // CHECK("tmp-var", "laplacian_nu_U[DIAG]", fvm::laplacian(nu, U).ref().diag());
        // CHECK("tmp-var", "laplacian_nu_U[SOURCE]", fvm::laplacian(nu, U).ref().source());
        // CHECK("tmp-var boundary", "laplacian_nu_U[BOUNDARYCOEFFS]", fvm::laplacian(nu, U).ref().boundaryCoeffs());
        // CHECK("tmp-var boundary 3->1", "laplacian_nu_U[INTERNALCOEFFS]", fvm::laplacian(nu, U).ref().internalCoeffs());

        CHECK("Eqn laplacian tmp-var", "laplacian_nu_U", fvm::laplacian(nu, U).ref());
        CHECK("Eqn", "UEqn", UEqn);

        print("make_U_momentumPredictor()");
        
        if (piso.momentumPredictor())
        {
            solve(UEqn == -fvc::grad(p));
        }

        CHECK("", "fvc_grad_p", fvc::grad(p).ref());
        CHECK("", "U", U);

        // --- PISO loop
        while (piso.correct())
        {
            print("make_pUqn()");

            volScalarField rAU(1.0/UEqn.A());
            volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));
            // PRINT(HbyA.boundaryField());
            // PRINT(U.boundaryField());
            // print2py(Info, U.boundaryField()[3]);
            // PRINT(phi.boundaryField());
            surfaceScalarField phiHbyA
            (
                "phiHbyA",
                fvc::flux(HbyA)
              + fvc::interpolate(rAU)*fvc::ddtCorr(U, phi)
            );
            // PRINT(fvc::flux(HbyA).ref().boundaryField());
            // PRINT(phiHbyA);
            adjustPhi(phiHbyA, U, p);

            // Update the pressure BCs to ensure flux consistency
            constrainPressure(p, U, phiHbyA, rAU);
            // PRINT(phiHbyA.boundaryField());

            // SET("boundary", phiHbyA.boundaryField());

            CHECK("tmp-var", "rAU", rAU);
            CHECK("tmp-var", "HbyA", HbyA);

            CHECK("tmp-var", "flux_HbyA", fvc::flux(HbyA).ref());
            CHECK("tmp-var", "interpolate_rAU", fvc::interpolate(rAU).ref());
            CHECK("tmp-var", "ddtCorr", fvc::ddtCorr(U, phi).ref());

            CHECK("tmp-var", "phiHbyA", phiHbyA);
            // CHECK("boundary", "HbyA_boundary", HbyA.boundaryField());
            // CHECK("boundary", "phiHbyA_boundaryField_", phiHbyA.boundaryField());

            // Non-orthogonal pressure corrector loop
            while (piso.correctNonOrthogonal())
            {
                // Pressure corrector

                fvScalarMatrix pEqn
                (
                    fvm::laplacian(rAU, p) == fvc::div(phiHbyA)
                );
                PRINT("laplacian(" + rAU.name() + ',' + p.name() + ')');
                
                // CHECK("tmp-var", "div_phiHbyA", fvc::div(phiHbyA).ref()); // 不能check
                CHECK("tmp-var laplacian Eqn", "pEqn", pEqn);
                auto tmp = pEqn.diag()[pRefCell];
                pEqn.setReference(pRefCell, pRefValue);
                if (pEqn.diag()[pRefCell] != tmp)
                {
                    print("setReference()");
                    CHECK("laplacian Eqn", "pEqn", pEqn);
                }
                pEqn.solve();

                print("slove_p()");
                CHECK("", "p", p);

                print("update_pu()");

                if (piso.finalNonOrthogonalIter())
                {
                    phi = phiHbyA - pEqn.flux();
                }

                CHECK("", "phi", phi);

            }

            #include "continuityErrs.H"

            U = HbyA - rAU*fvc::grad(p);

            CHECK("", "fvc_grad_p", fvc::grad(p).ref());
            // CHECK("", "U", U);

            U.correctBoundaryConditions();
            CHECK("", "U", U);
        }

        runTime.write();

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }

    Info<< "End\n" << endl;

    Info << "'''" << endl;
    return 0;
}


// ************************************************************************* //
