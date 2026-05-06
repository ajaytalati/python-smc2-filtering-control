# tools/_RunDir.jl — atomic run-directory allocator.
#
# Port of `version_3/tools/_run_dir.py`. Same TOCTOU-race fix: the
# allocator uses `mkdir(...)` with `exist_ok=false` semantics in a
# bounded retry loop so two concurrent processes can't end up with
# the same numbered directory.

module RunDir

using Printf

const _MAX_ALLOC_RETRIES = 256

"""
    scan_max_run_number(experiments_dir::AbstractString)::Int

Return the largest existing run number under `experiments_dir`, or 0
if there are none. Expects directories of the form `runNN_<tag>`.
"""
function scan_max_run_number(experiments_dir::AbstractString)::Int
    isdir(experiments_dir) || return 0
    nums = Int[]
    for entry in readdir(experiments_dir)
        full = joinpath(experiments_dir, entry)
        if isdir(full) && startswith(entry, "run")
            stem = split(entry[4:end], "_"; limit=2)[1]
            n = tryparse(Int, stem)
            n !== nothing && push!(nums, n)
        end
    end
    return isempty(nums) ? 0 : maximum(nums)
end

"""
    allocate_run_dir(repo_root::AbstractString, run_tag::AbstractString)::Tuple{String,Int}

Atomically reserve `outputs/fsa_v5/experiments/runNN_<run_tag>/`.
Returns `(out_dir, NN)`. Two concurrent processes both pass the
existence check via the bounded retry loop with `mkdir` (Julia's
`mkdir` raises if the dir already exists — that's our atomic flag).

Mirrors `_run_dir.allocate_run_dir` in Python.
"""
function allocate_run_dir(repo_root::AbstractString,
                          run_tag::AbstractString)::Tuple{String,Int}
    exp_dir = joinpath(repo_root, "outputs", "fsa_v5", "experiments")
    mkpath(exp_dir)

    last_err = nothing
    for attempt in 0:_MAX_ALLOC_RETRIES-1
        n = scan_max_run_number(exp_dir) + 1 + attempt
        out_dir = joinpath(exp_dir, "run" * (@sprintf "%02d" n) * "_" * run_tag)
        try
            mkdir(out_dir)
            return out_dir, n
        catch err
            last_err = err
            # Race: another process took this number; bump and retry.
            continue
        end
    end
    error("Could not allocate a run dir after $_MAX_ALLOC_RETRIES " *
          "attempts under $exp_dir. Last error: $last_err")
end

export scan_max_run_number, allocate_run_dir

end # module RunDir
