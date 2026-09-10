llm_name=$1
test_ratio=${2:-500}

echo "vanilla"
python inference.py --llm_name "$llm_name" --prop 'LogP'                         --test_ratio "$test_ratio" --accuracy_only
python inference.py --llm_name "$llm_name" --prop 'TPSA'                         --test_ratio "$test_ratio" --accuracy_only
python inference.py --llm_name "$llm_name" --prop 'QED'                          --test_ratio "$test_ratio" --accuracy_only
python inference.py --llm_name "$llm_name" --prop 'LogP&TPSA'                    --test_ratio "$test_ratio" --accuracy_only
python inference.py --llm_name "$llm_name" --prop 'LogP&QED'                     --test_ratio "$test_ratio" --accuracy_only
echo "ICL"
python inference.py --llm_name "$llm_name" --prop 'LogP'      --icl              --test_ratio "$test_ratio" --accuracy_only
python inference.py --llm_name "$llm_name" --prop 'TPSA'      --icl              --test_ratio "$test_ratio" --accuracy_only
python inference.py --llm_name "$llm_name" --prop 'QED'       --icl              --test_ratio "$test_ratio" --accuracy_only
python inference.py --llm_name "$llm_name" --prop 'LogP&TPSA' --icl              --test_ratio "$test_ratio" --accuracy_only
python inference.py --llm_name "$llm_name" --prop 'LogP&QED'  --icl              --test_ratio "$test_ratio" --accuracy_only
echo "CoT-Lite"
python inference.py --llm_name "$llm_name" --prop 'LogP'      --icl --cot --lite --test_ratio "$test_ratio" --accuracy_only
python inference.py --llm_name "$llm_name" --prop 'TPSA'      --icl --cot --lite --test_ratio "$test_ratio" --accuracy_only
python inference.py --llm_name "$llm_name" --prop 'QED'       --icl --cot --lite --test_ratio "$test_ratio" --accuracy_only
python inference.py --llm_name "$llm_name" --prop 'LogP&TPSA' --icl --cot --lite --test_ratio "$test_ratio" --accuracy_only
python inference.py --llm_name "$llm_name" --prop 'LogP&QED'  --icl --cot --lite --test_ratio "$test_ratio" --accuracy_only
# echo "CoT"
# python inference.py --llm_name "$llm_name" --prop 'LogP'      --icl --cot        --test_ratio "$test_ratio" --accuracy_only
# python inference.py --llm_name "$llm_name" --prop 'TPSA'      --icl --cot        --test_ratio "$test_ratio" --accuracy_only
# python inference.py --llm_name "$llm_name" --prop 'QED'       --icl --cot        --test_ratio "$test_ratio" --accuracy_only
# python inference.py --llm_name "$llm_name" --prop 'LogP&TPSA' --icl --cot        --test_ratio "$test_ratio" --accuracy_only
# python inference.py --llm_name "$llm_name" --prop 'LogP&QED'  --icl --cot        --test_ratio "$test_ratio" --accuracy_only