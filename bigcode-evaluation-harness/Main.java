class Solution {
    public int findFirstDuplicate(int[] nums) {
        HashSet<Integer> numSet = new HashSet<>();
        int noDuplicate = -1;
        for (int i = 0; i < nums.length; i = i + 1) {
            if (numSet.contains(nums[i])) {
                return nums[i];
            } else {
                numSet.add(nums[i]);
            }
        }
        return noDuplicate;
    }
}