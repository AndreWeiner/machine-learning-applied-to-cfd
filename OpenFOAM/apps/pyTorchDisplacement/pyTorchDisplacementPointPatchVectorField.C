/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2004-2010 OpenCFD Ltd.
     \\/     M anipulation  |
-------------------------------------------------------------------------------
                            | Copyright (C) 2011-2016 OpenFOAM Foundation
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

\*---------------------------------------------------------------------------*/

#include "pyTorchDisplacementPointPatchVectorField.H"
#include "pointPatchFields.H"
#include "addToRunTimeSelectionTable.H"
#include "Time.H"
#include "polyMesh.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

pyTorchDisplacementPointPatchVectorField::
pyTorchDisplacementPointPatchVectorField
(
    const pointPatch& p,
    const DimensionedField<vector, pointMesh>& iF
)
:
    fixedValuePointPatchField<vector>(p, iF),
    center_(Zero),
    model_name_("shape_model.pt")
{}


pyTorchDisplacementPointPatchVectorField::
pyTorchDisplacementPointPatchVectorField
(
    const pointPatch& p,
    const DimensionedField<vector, pointMesh>& iF,
    const dictionary& dict
)
:
    fixedValuePointPatchField<vector>(p, iF, dict),
    center_(dict.lookup("center")),
    model_name_(dict.lookupOrDefault<word>("model", "shape_model.pt"))
{
    pyTorch_model_ = torch::jit::load(model_name_);
    assert(pyTorch_model_ != nullptr);
    if (!dict.found("value"))
    {
        updateCoeffs();
    }
}


pyTorchDisplacementPointPatchVectorField::
pyTorchDisplacementPointPatchVectorField
(
    const pyTorchDisplacementPointPatchVectorField& ptf,
    const pointPatch& p,
    const DimensionedField<vector, pointMesh>& iF,
    const pointPatchFieldMapper& mapper
)
:
    fixedValuePointPatchField<vector>(ptf, p, iF, mapper),
    center_(ptf.center_),
    model_name_(ptf.model_name_),
    pyTorch_model_(ptf.pyTorch_model_)
{}


pyTorchDisplacementPointPatchVectorField::
pyTorchDisplacementPointPatchVectorField
(
    const pyTorchDisplacementPointPatchVectorField& ptf,
    const DimensionedField<vector, pointMesh>& iF
)
:
    fixedValuePointPatchField<vector>(ptf, iF),
    center_(ptf.center_),
    model_name_(ptf.model_name_),
    pyTorch_model_(ptf.pyTorch_model_)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void pyTorchDisplacementPointPatchVectorField::updateCoeffs()
{
    if (this->updated())
    {
        return;
    }

    const polyMesh& mesh = this->internalField().mesh()();
    const Time& t = mesh.time();
    const pointField& localPoints = patch().localPoints();
    torch::Tensor featureTensor = torch::ones({localPoints.size(), 3});

    forAll(localPoints, i)
    {
        scalar pi = constant::mathematical::pi;
        vector x = localPoints[i] - center_;
        scalar r = sqrt(x & x);
        scalar phi = acos(x.y() / r);
        scalar theta = std::fmod((atan2(x.x(), x.z()) + pi), pi);
        if (x.x() < 0.0)
        {
            phi = 2.0 * pi - phi;
        }
        featureTensor[i][0] = phi;
        featureTensor[i][1] = theta;
        featureTensor[i][2] = t.value();
    }

    std::vector<torch::jit::IValue> modelFeatures{featureTensor};
    torch::Tensor radTensor = pyTorch_model_->forward(modelFeatures).toTensor();
    auto radAccessor = radTensor.accessor<float,1>();
    vectorField result(localPoints.size(), Zero);
    forAll(result, i)
    {
        vector x = localPoints[i] - center_;
        result[i] = x / mag(x) * (radAccessor[i] - mag(x));
    }


    Field<vector>::operator=(result);

    fixedValuePointPatchField<vector>::updateCoeffs();
}


void pyTorchDisplacementPointPatchVectorField::write(Ostream& os) const
{
    pointPatchField<vector>::write(os);
    os.writeEntry("center", center_);
    os.writeEntry("model_name", model_name_);
    writeEntry("value", os);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

makePointPatchTypeField
(
    pointPatchVectorField,
    pyTorchDisplacementPointPatchVectorField
);

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
