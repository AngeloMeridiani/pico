#!/bin/bash
# Trap SIGINT (Ctrl+C) and call cleanup function
trap cleanup SIGINT

[[ -n "$SLURM_PARAMS" ]] && inform "Sbatch params:" "$SLURM_PARAMS"

####################################################################################
#                           MAIN BENCHMARKING LOOP                                 #
####################################################################################
# WARN: The division between tui loop and normal loop is temporary, it will be changed
if [[ -d "$TUI_FILE" ]]; then
    iter=0
    for config in ${TEST_CONFIG_FILES[@]//,/ }; do
        export TEST_CONFIG=${config}
        export TEST_ENV="${TEST_CONFIG}_env.sh"

        # Now --gpu-per-node is analyzed. It is a comma separated list of GPUs per node.
        # We need to iterate through it and set the GPU_AWARENESS variable accordingly.
        # When a 0 is found, the GPU_AWARENESS is set to no and the test is run on CPU,
        # iterating through the --tasks-per-node list.
        for n_gpu in ${GPU_PER_NODE[@]//,/ }; do
            if [[ "$n_gpu" == "0" ]]; then
                export GPU_AWARENESS="no"
                export PICO_EXEC=$PICO_EXEC_CPU

                for ntasks in ${TASKS_PER_NODE[@]//,/ }; do
                    export CURRENT_TASKS_PER_NODE=$ntasks
                    export MPI_TASKS=$(( N_NODES * CURRENT_TASKS_PER_NODE ))

                    # --ntasks will override any --tasks-per-node value,
                    # CURRENT_TASKS_PER_NODE is set for metadata reasons
                    # and is a truncated value not representative of actual allocation.
                    if [[ -n "$FORCE_TASKS" ]]; then
                        export MPI_TASKS=$FORCE_TASKS
                        export CURRENT_TASKS_PER_NODE=$(( FORCE_TASKS / N_NODES ))
                    fi

                    # Run script to parse and generate test environment variables
                    python3 $BINE_DIR/config/parse_test.py || exit 1
                    source $TEST_ENV
                    load_other_env_var
                    success "📄 Test configuration ${TEST_CONFIG} parsed (CPU, ntasks=${CURRENT_TASKS_PER_NODE})"

                    # Create the metadata if not in debug mode or dry run
                    if [[ "$DEBUG_MODE" == "no" && "$DRY_RUN" == "no" ]]; then
                        export DATA_DIR="$OUTPUT_DIR/$iter"
                        mkdir -p "$DATA_DIR"
                        python3 $BINE_DIR/results/generate_metadata.py $iter || exit 1
                        success "📂 Metadata of $DATA_DIR created"
                    fi

                    print_sanity_checks

                    # Run the tests
                    run_all_tests
                    ((iter++))

                    # If --ntasks is set, we skip the possible --tasks-per-node values
                    if [[ -n "$FORCE_TASKS" ]]; then
                        warning "--ntasks is set, skipping possible --tasks-per-node values"
                        break
                    fi
                done
            else
                export GPU_AWARENESS="yes"
                export CURRENT_TASKS_PER_NODE=$n_gpu
                export MPI_TASKS=$(( N_NODES * n_gpu ))
                export PICO_EXEC=$PICO_EXEC_GPU

                # Run script to parse and generate test environment variables
                python3 $BINE_DIR/config/parse_test.py || exit 1
                source $TEST_ENV
                load_other_env_var
                success "📄 Test configuration ${TEST_CONFIG} parsed (GPU, gpus per node=${CURRENT_TASKS_PER_NODE})"

                # Create the metadata if not in debug mode or dry run
                if [[ "$DEBUG_MODE" == "no" && "$DRY_RUN" == "no" ]]; then
                    export DATA_DIR="$OUTPUT_DIR/$iter"
                    mkdir -p "$DATA_DIR"
                    python3 $BINE_DIR/results/generate_metadata.py $iter || exit 1
                    success "📂 Metadata of $DATA_DIR created"
                fi

                print_sanity_checks

                # Run the tests
                run_all_tests
                ((iter++))
            fi
        done
    done
else
    iter=0
    for coll in ${COLLECTIVES//,/ }; do
        COLL_LOWER="${coll,,}"         # e.g., allreduce
        COLL_UPPER="${coll^^}"         # e.g., ALLREDUCE

        # --- Backward-compat layer per collective ---
        # 1) COLLECTIVE_TYPE (UPPERCASE)
        export COLLECTIVE_TYPE="${COLL_UPPER}"

        # 2) ALGOS (space-separated from ${COLL_UPPER}_ALGORITHMS which is comma-separated)
        ALGS_VAR="${COLL_UPPER}_ALGORITHMS"
        ALGS_CSV="$(_get_var "$ALGS_VAR")"
        if [[ -z "$ALGS_CSV" ]]; then
            warning "No algorithms found for collective ${coll} (${ALGS_VAR} is empty). Skipping."
            continue
        fi
        # Split into bash array of names (no spaces in names expected)
        IFS=',' read -r -a _ALG_NAMES <<< "$ALGS_CSV"
        # Space-separated for legacy ALGOS var
        export ALGOS="${ALGS_CSV//,/ }"

        # 3) SKIP (space-separated from ${COLL_UPPER}_ALGORITHMS_SKIP which is comma-separated)
        SKIPS_VAR="${COLL_UPPER}_ALGORITHMS_SKIP"
        SKIPS_CSV="$(_get_var "$SKIPS_VAR")"
        # Split into bash array of flags (no spaces in names expected)
        IFS=',' read -r -a _SKIP_FLAGS <<< "$SKIPS_CSV"
        # Space-separated for legacy SKIP var
        export SKIP="${SKIPS_CSV//,/ }"

        # 4) IS_SEGMENTED (take from ${coll}_Algorithms_is_segmented[@] when available; else all "no")
        #    Note the exact requested name uses the *lower-case* key and mixed-case suffix.
        SEG_ARR_VAR="${COLL_UPPER}_ALGORITHMS_IS_SEGMENTED"
        SEG_CSV="$(_get_var "$SEG_ARR_VAR")"
        if [[ -z "$SEG_CSV" ]]; then
            # Fallback: all "no" with same length as _ALG_NAMES
            warning "No segmentation flags found for collective ${coll} (${SEG_ARR_VAR} is empty). Using all 'no'."
            IS_SEGMENTED=()
            for ((i=0; i<${#_ALG_NAMES[@]}; i++)); do IS_SEGMENTED+=(no); done
        else
            IFS=',' read -r -a IS_SEGMENTED <<< "$SEG_CSV"
            # Sanity: ensure same length as algorithms (pad/trim and warn)
            if (( ${#IS_SEGMENTED[@]} < ${#_ALG_NAMES[@]} )); then
                warning "IS_SEGMENTED shorter than algorithms for ${coll}; padding with 'no'."
                for ((i=${#IS_SEGMENTED[@]}; i<${#_ALG_NAMES[@]}; i++)); do IS_SEGMENTED+=(no); done
            elif (( ${#IS_SEGMENTED[@]} > ${#_ALG_NAMES[@]} )); then
                warning "IS_SEGMENTED longer than algorithms for ${coll}; trimming extras."
                IS_SEGMENTED=( "${IS_SEGMENTED[@]:0:${#_ALG_NAMES[@]}}" )
            fi
        fi

        # 5) if MPI_LIB is CRAY_MPICH or MPICH create a CVAR

        # Now --gpu-per-node is analyzed. It is a comma separated list of GPUs per node.
        # We need to iterate through it and set the GPU_AWARENESS variable accordingly.
        # When a 0 is found, the GPU_AWARENESS is set to no and the test is run on CPU,
        # iterating through the --tasks-per-node list.
        for n_gpu in ${GPU_PER_NODE[@]//,/ }; do
            if [[ "$n_gpu" == "0" ]]; then
                export GPU_AWARENESS="no"
                export PICO_EXEC=$PICO_EXEC_CPU

                for ntasks in ${TASKS_PER_NODE[@]//,/ }; do
                    export CURRENT_TASKS_PER_NODE=$ntasks
                    export MPI_TASKS=$(( N_NODES * CURRENT_TASKS_PER_NODE ))

                    # --ntasks will override any --tasks-per-node value,
                    # CURRENT_TASKS_PER_NODE is set for metadata reasons
                    # and is a truncated value not representative of actual allocation.
                    if [[ -n "$FORCE_TASKS" ]]; then
                        export MPI_TASKS=$FORCE_TASKS
                        export CURRENT_TASKS_PER_NODE=$(( FORCE_TASKS / N_NODES ))
                    fi

                    # Run script to parse and generate test environment variables
                    load_other_env_var
                    success "📄 Test configuration ${TEST_CONFIG} parsed (CPU, ntasks=${CURRENT_TASKS_PER_NODE})"

                    # Create the metadata if not in debug mode or dry run
                    if [[ "$DEBUG_MODE" == "no" && "$DRY_RUN" == "no" ]]; then
                        export DATA_DIR="$OUTPUT_DIR/$iter"
                        mkdir -p "$DATA_DIR"
                        python3 $BINE_DIR/results/generate_metadata.py $iter || exit 1
                        success "📂 Metadata of $DATA_DIR created"
                    fi

                    print_sanity_checks

                    # Run the tests
                    run_all_tests
                    ((iter++))

                    # If --ntasks is set, we skip the possible --tasks-per-node values
                    if [[ -n "$FORCE_TASKS" ]]; then
                        warning "--ntasks is set, skipping possible --tasks-per-node values"
                        break
                    fi
                done
            else
                export GPU_AWARENESS="yes"
                export CURRENT_TASKS_PER_NODE=$n_gpu
                export MPI_TASKS=$(( N_NODES * n_gpu ))
                export PICO_EXEC=$PICO_EXEC_GPU

                # Run script to parse and generate test environment variables
                load_other_env_var
                success "📄 Test configuration ${TEST_CONFIG} parsed (GPU, gpus per node=${CURRENT_TASKS_PER_NODE})"

                # Create the metadata if not in debug mode or dry run
                if [[ "$DEBUG_MODE" == "no" && "$DRY_RUN" == "no" ]]; then
                    export DATA_DIR="$OUTPUT_DIR/$iter"
                    mkdir -p "$DATA_DIR"
                    python3 $BINE_DIR/results/generate_metadata.py $iter || exit 1
                    success "📂 Metadata of $DATA_DIR created"
                fi

                print_sanity_checks

                # Run the tests
                run_all_tests
                ((iter++))
            fi
        done
    done
fi

success "All tests completed successfully"

###################################################################################
#              COMPRESS THE RESULTS AND DELETE THE OUTPUT DIR IF REQUESTED        #
###################################################################################
if [[ "$DEBUG_MODE" == "no" && "$DRY_RUN" == "no" && "$COMPRESS" == "yes" ]]; then
    tarball_path="$(dirname "$OUTPUT_DIR")/$(basename "$OUTPUT_DIR").tar.gz"
    if tar -czf "$tarball_path" -C "$(dirname "$OUTPUT_DIR")" "$(basename "$OUTPUT_DIR")"; then
        if [[ "$DELETE" == "yes" ]]; then
            rm -rf "$OUTPUT_DIR"
        fi
    fi
fi

squeue -j $SLURM_JOB_ID

