using JLD2

"""
Load all .jld2 files and return as vector of named tuples
Each tuple contains filename and data
"""
function load_all_jld2_vector(folder_path::String)
    jld2_files = filter(f -> endswith(f, ".jld2"), readdir(folder_path))
    
    results = []
    
    for file in jld2_files
        filepath = joinpath(folder_path, file)
        
        try
            data = load(filepath)
            push!(results, (filename=file, data=data))
        catch e
            println("Error loading $file: $e")
        end
    end
    
    return results
end

# # Usage
# all_files = load_all_jld2_vector(".")

# for item in all_files
#     println("File: $(item.filename)")
#     println("  Data: ", keys(item.data))
# end