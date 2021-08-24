from inspect import getmembers, isfunction, getargspec
import basenji.blocks
import json

m = [(x[0], getargspec(x[1])) for x in getmembers(basenji.blocks, isfunction)]

out = []

for function in m:
    entry = {}
    entry["params"] = []
    argspec = function[1]
    
    # names
    rev_args = argspec.args[::-1]
    
    # values
    print(type(argspec.defaults))
    print(function[0])
    rev_defaults = argspec.defaults[::-1] if argspec.defaults is not None else []

    for n, x in enumerate(rev_defaults):
        entry["params"].append({
            "name":rev_args[n],
            "value":x
        })

    entry["kwargs"] = argspec.keywords
    entry["function_name"] = function[0]

    out.append(entry)
    # out[function[0]] = function[1]

# print(out["conv_dna"])


file = open("params_react.json", "w")
file.write(json.dumps(out, indent=4, sort_keys=True))
