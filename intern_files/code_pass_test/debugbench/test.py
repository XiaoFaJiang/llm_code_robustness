from evaluate import load
from datasets import load_from_disk
import os
import json
import re
os.environ['HF_ALLOW_CODE_EVAL']= '1'
os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:/usr/local/lib64'

code_metric = load("./bigcode-evaluation-harness/bigcode_eval/tasks/custom_metrics/code_eval_octopack")


if __name__ == '__main__':
    code_str = '''
function remove_Occ(s, ch) {
    for (let i = 0; i < s.length; i++) {
        if (s[i] === ch) {
            s = s.substring(0, i) + s.substring(i + 1);
            break;
        }
    }
    return s;
}

    '''
    test = '''
const testRemove_Occ = () => {

    console.assert(remove_Occ("", "l") === "", "Test 5 failed");
}
testRemove_Occ() // invoke test
'''

    ans = []
    metrics, cases = code_metric.compute(
            references=[test],
            predictions=[[code_str]],
            language='javascript',
            timeout=60.0,
            num_workers=4,
    )
    print(cases)
    print(metrics)




