llm_name=$1
test_ratio=${2:-500}

python inference.py --llm_name "$llm_name" --prop 'LogP'      --icl --test_ratio "$test_ratio" --accuracy_only
python inference.py --llm_name "$llm_name" --prop 'TPSA'      --icl --test_ratio "$test_ratio" --accuracy_only
python inference.py --llm_name "$llm_name" --prop 'QED'       --icl --test_ratio "$test_ratio" --accuracy_only
python inference.py --llm_name "$llm_name" --prop 'LogP&TPSA' --icl --test_ratio "$test_ratio" --accuracy_only
python inference.py --llm_name "$llm_name" --prop 'LogP&QED'  --icl --test_ratio "$test_ratio" --accuracy_only