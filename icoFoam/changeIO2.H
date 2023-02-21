#include "fvCFD.H"

template <typename _Ty>
Ostream &printNone(Ostream &os, _Ty &&x, bool isNone);

#define DEC_PRINT2PY_FORWARD(type)                     \
    Foam::Ostream &print2py(Foam::Ostream &os, type v) \
    {                                                  \
        return os << v;                                \
    }

Foam::Ostream &print2py(Foam::Ostream &os, const Foam::word &v)
{
    return os << "'" << v << "'";
}

Ostream &print2py(Ostream &os, const scalar v)
{
    constexpr auto hexfloat = ios_base::fixed | ios_base::scientific;
    if ((os.flags() & hexfloat) == hexfloat)
        return os << '\'' << v << '\'';
    return os << v;
}

template <typename Type, typename std::enable_if<!std::is_class<typename std::decay<Type>::type>::value, int>::type = 1>
Ostream &print2py(Ostream &os, Type &&v)
{
    return os << std::forward<Type>(v);
}

template <class Form, class Cmpt, direction Ncmpts>
Ostream &print2py(Ostream &os, const VectorSpace<Form, Cmpt, Ncmpts> &list)
{

    os << "(";
    const char *delim = "";
    for (Foam::label i = 0; i < list.nComponents; ++i)
    {

        os << delim;
        delim = ", ";
        print2py(os, list[i]);
    }
    os << ")";
    return os;
}


template <typename ListType>
Ostream &print2pylist(Ostream &os, const ListType &list);

template <typename Type, typename std::enable_if<!std::is_void<typename Type::iterator>::value, int>::type = 1>
Ostream &print2py(Ostream &os, const Type &list)
{
    return print2pylist(os, list);
}


template <typename Type>
Ostream &print2py(Ostream &os, const tmp<Type> &tmpv)
{
    return print2py(tmpv.ref());
}

template <typename Type>
Ostream &print2py(Ostream &os, const fvMatrix<Type> &fvm)
{

    os << "[";
    (fvm.hasLower() ? print2pylist(os, fvm.lower()) : os << "0") << ",\n";
    (fvm.hasDiag() ? print2pylist(os, fvm.diag()) : os << "0") << ",\n";
    (fvm.hasUpper() ? print2pylist(os, fvm.upper()) : os << "0") << ",\n";
    print2py(os, fvm.source()) << ",\n";
    auto &rfvm = const_cast<fvMatrix<Type> &>(fvm);
    print2py(os, rfvm.internalCoeffs()) << ",\n";
    print2py(os, rfvm.boundaryCoeffs());
    os << "]";

    os.check("pyOstream& operator<<(pyOstream&, fvMatrix<Type>&");

    return os;
}


template <typename ListType>
Ostream &print2pylist(Ostream &os, const ListType &list)
{
    os << "[";
    const char *delim = "";
    for (auto &x: list)
    {
        
        os << delim;
        delim = ", ";
        print2py(os, x);
    }
    os << "]";
    return os;
}


// template<class Type, template<class> class PatchField, class GeoMesh>
// Ostream &print2py(Ostream &os, const Foam::GeometricField<Type, PatchField, GeoMesh> &_GeometricField)
// {
//     os << "[";
//     print2py(os, _GeometricField.internalField()) << ",\n";
//     print2py(os, _GeometricField.boundaryField()) << "\n]})";
//     return os;
// }



// template <typename Type>
// Ostream &print2py(Ostream &os, const Type &x)
// {
//     return os <<  "\"\"\"\n" << x << "\n\"\"\"";
// }
