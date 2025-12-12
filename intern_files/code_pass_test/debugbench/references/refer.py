import re
import json
def get_reference(doc):
        parts = doc["examples"]
        gt = doc["oracle_code"]
        name = gt.split('(')[0].strip().split(' ')[-1]
        output_pattern = re.compile(r"Output: ([^\n]+)")
        fun = ""
        for item in parts:    
            outputs = output_pattern.findall(item)

            if len(outputs) == 1:
                output = outputs[0]
            else:
                output = "(" + ", ".join(outputs) + ")"

            output = output.replace("[","{").replace("]","}")
        
        
            s = item.split("Output:")[0]
            parts = s.split('=')
            l = [part.rsplit(',', 1)[0].strip() for part in parts[1:-1]]
            l.append(parts[-1].strip().rstrip('\n'))
            l = ', '.join(str(result) for result in l)
            ans = l.replace("[","(").replace("]",")")
            fun += f"s.{name}({ans}) == {output},\n"
        fun = fun.replace("false", "False")
        fun = fun.replace("true", "True")
        fun = fun.replace("null", "None")
        fun = fun.rstrip(",\n")
        head = """
public class Main {
    public static void main(String[] args) {
        Solution s = new Solution();
        List<Boolean> correct = Arrays.asList(
"""
        tail = """
        );
        if (correct.contains(false)) {
            throw new AssertionError();
        }
    }
}
""" 
        fun = head + fun + tail
        return fun

list = []
with open("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zhaoyicong03/dataset/flitered_java.json",'r') as file:
    data = json.load(file)
    for item in data:
        list.append(get_reference(item))

filename = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zhaoyicong03/references/java_references.json'

# 使用 'with' 语句来打开文件，确保它会被正确地关闭
with open(filename, 'w') as file:
    # 使用 json.dump 将数据写入文件
    json.dump(list, file)

print(len(list))

# data = {
#         "slug": "relative-sort-array",
#         "description": "Given two arrays arr1 and arr2, the elements of arr2 are distinct, and all elements in arr2 are also in arr1.\nSort the elements of arr1 such that the relative ordering of items in arr1 are the same as in arr2. Elements that do not appear in arr2 should be placed at the end of arr1 in ascending order.",
#         "examples": [
#             "Input: arr1 = [2,3,1,3,2,4,6,7,9,2,19], arr2 = [2,1,4,3,9,6]\nOutput: [2,2,2,1,4,3,3,9,6,7,19]",
#             "Input: arr1 = [28,6,22,8,44,17], arr2 = [22,28,8,6]\nOutput: [22,28,8,6,17,44]"
#         ],
#         "constraints": "1 <= arr1.length, arr2.length <= 1000\n0 <= arr1[i], arr2[i] <= 1000\nAll the elements of arr2 are distinct.\nEach\u00a0arr2[i] is in arr1.",
#         "release_time": 1691725804,
#         "oracle_code": "import java.util.Arrays;\n\nclass Solution {\n    public int[] relativeSortArray(int[] arr1, int[] arr2) {\n        Arrays.sort(arr1);\n        int[] ans = new int[arr1.length];\n        int start = 0;\n        int end = arr1.length - 1;\n        int index = 0;\n\n        for (int i = 0; i < arr2.length; i++) {\n            int target = arr2[i];\n            start = 0;\n            end = arr1.length - 1;\n\n            while (end >= start) {\n                int mid = start + (end - start) / 2;\n                if (arr1[mid] == target) {\n                    for (int j = start; j <= end; j++) {\n                        if (arr1[j] == target) {\n                            ans[index++] = arr1[j];\n                        }\n                    }\n                    break;\n                }\n                if (arr1[mid] < target) {\n                    start = mid + 1;\n                } else {\n                    end = mid - 1;\n                }\n            }\n        }\n\n        for (int i = 0; i < arr1.length; i++) {\n            if (index == arr1.length) {\n                break;\n            }\n            boolean found = false;\n            for (int num : arr2) {\n                if (arr1[i] == num) {\n                    found = true;\n                    break;\n                }\n            }\n            if (!found) {\n                ans[index++] = arr1[i];\n            }\n        }\n\n        return ans;\n    }\n}",
#         "content": "# Intuition\\n<!-- Describe your first thoughts on how to solve this problem. -->\\n\\n# Approach\\n<!-- Describe your approach to solving the problem. -->\\n\\n# Complexity\\n- Time complexity:\\n<!-- Add your time complexity here, e.g. $$O(n)$$ -->\\n\\n- Space complexity:\\n<!-- Add your space complexity here, e.g. $$O(n)$$ -->\\n\\n# Code\\n```\\nimport java.util.Arrays;\\n\\nclass Solution {\\n    public int[] relativeSortArray(int[] arr1, int[] arr2) {\\n        Arrays.sort(arr1);\\n        int[] ans = new int[arr1.length];\\n        int start = 0;\\n        int end = arr1.length - 1;\\n        int index = 0;\\n\\n        for (int i = 0; i < arr2.length; i++) {\\n            int target = arr2[i];\\n            start = 0;\\n            end = arr1.length - 1;\\n\\n            while (end >= start) {\\n                int mid = start + (end - start) / 2;\\n                if (arr1[mid] == target) {\\n                    for (int j = start; j <= end; j++) {\\n                        if (arr1[j] == target) {\\n                            ans[index++] = arr1[j];\\n                        }\\n                    }\\n                    break;\\n                }\\n                if (arr1[mid] < target) {\\n                    start = mid + 1;\\n                } else {\\n                    end = mid - 1;\\n                }\\n            }\\n        }\\n\\n        for (int i = 0; i < arr1.length; i++) {\\n            if (index == arr1.length) {\\n                break;\\n            }\\n            boolean found = false;\\n            for (int num : arr2) {\\n                if (arr1[i] == num) {\\n                    found = true;\\n                    break;\\n                }\\n            }\\n            if (!found) {\\n                ans[index++] = arr1[i];\\n            }\\n        }\\n\\n        return ans;\\n    }\\n}\\n\\n```",
#         "level": "easy",
#         "buggy_code": "\nimport java.util.Arrays;\n\nclass Solution {\n    public int[] relativeSortArray(int[] arr1, int[] arr2) {\n        Arrays.sort(arr1);\n        int[] ans = new int[arr1.length];\n        int start = 0;\n        int end = arr1.length - 1;\n        int index = 0;\n\n        for (int i = 0; i < arr2.length; i++) {\n            int target = arr2[i];\n            start = 0;\n            end = arr1.length - 1;\n\n            while (end >= start) {\n                int mid = start + (end - start) / 2;\n                if (arr1[mid] == target) {\n                    for (int j = start; j <= end; j++) {\n                        if (arr1[j] == target) {\n                            ans[index++] = arr1[j];\n                            arr1[j] = -1;\n                        }\n                    }\n                    break;\n                }\n                if (arr1[mid] < target) {\n                    start = mid + 1;\n                } else {\n                    end = mid - 1;\n                }\n            }\n        }\n\n        Arrays.sort(arr1);\n\n        for (int i = 0; i < arr1.length; i++) {\n            if (index == arr1.length) {\n                break;\n            }\n            boolean found = false;\n            for (int num : arr2) {\n                if (arr1[i] == num) {\n                    found = true;\n                    break;\n                }\n            }\n            if (!found && arr1[i] != -1) {\n                ans[index++] = arr1[i];\n            }\n        }\n\n        return ans;\n    }\n}\n",
#         "explanations": "\nThe bug I added was modifying the array in the sorting step without considering its impact on forthcoming operations, causing incorrect output.\n",
#         "source": "java_operation error.json"
#     }

# print(get_reference(data))