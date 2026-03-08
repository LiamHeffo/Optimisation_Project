-- modified gas model made by PITOT3 on the 17/07/2025 13:07:35 without ions for low temperature operation

model = "CEAGas"

CEAGas = {
  mixtureName = "he-with-ions-without-ions",
  speciesList = {'He'},
  reactants = {He = 1.0},
  inputUnits = 'moles',
  withIons = false,
  trace = 1e-06
}