#include <iostream>
#include <queue>
#include <vector>

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

// Inorder traversal (Left, Root, Right)
void inorder(TreeNode* root) {
    if (!root) return;
    inorder(root->left);
    std::cout << root->val << " ";
    inorder(root->right);
}

// Preorder traversal (Root, Left, Right)
void preorder(TreeNode* root) {
    if (!root) return;
    std::cout << root->val << " ";
    preorder(root->left);
    preorder(root->right);
}

// Postorder traversal (Left, Right, Root)
void postorder(TreeNode* root) {
    if (!root) return;
    postorder(root->left);
    postorder(root->right);
    std::cout << root->val << " ";
}

// Level order traversal (BFS)
void levelOrder(TreeNode* root) {
    if (!root) return;
    std::queue<TreeNode*> q;
    q.push(root);
    while (!q.empty()) {
        TreeNode* node = q.front();
        q.pop();
        std::cout << node->val << " ";
        if (node->left) q.push(node->left);
        if (node->right) q.push(node->right);
    }
}

// Interview: Max depth
int maxDepth(TreeNode* root) {
    if (!root) return 0;
    return 1 + std::max(maxDepth(root->left), maxDepth(root->right));
}

// Interview: Is valid BST
bool isValidBSTHelper(TreeNode* root, long long minVal, long long maxVal) {
    if (!root) return true;
    if (root->val <= minVal || root->val >= maxVal) return false;
    return isValidBSTHelper(root->left, minVal, root->val) &&
           isValidBSTHelper(root->right, root->val, maxVal);
}

bool isValidBST(TreeNode* root) {
    return isValidBSTHelper(root, LLONG_MIN, LLONG_MAX);
}

// Interview: Lowest common ancestor
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    if (!root || root == p || root == q) return root;
    TreeNode* left = lowestCommonAncestor(root->left, p, q);
    TreeNode* right = lowestCommonAncestor(root->right, p, q);
    if (left && right) return root;
    return left ? left : right;
}

int main() {
    TreeNode* root = new TreeNode(4);
    root->left = new TreeNode(2);
    root->right = new TreeNode(6);
    root->left->left = new TreeNode(1);
    root->left->right = new TreeNode(3);
    root->right->left = new TreeNode(5);
    root->right->right = new TreeNode(7);
    
    std::cout << "Inorder: "; inorder(root); std::cout << std::endl;
    std::cout << "Preorder: "; preorder(root); std::cout << std::endl;
    std::cout << "Postorder: "; postorder(root); std::cout << std::endl;
    std::cout << "Level order: "; levelOrder(root); std::cout << std::endl;
    std::cout << "Max depth: " << maxDepth(root) << std::endl;
    std::cout << "Is valid BST: " << (isValidBST(root) ? "Yes" : "No") << std::endl;
    
    return 0;
}
