#include <iostream>

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

// Interview: Reverse linked list
ListNode* reverseList(ListNode* head) {
    ListNode* prev = nullptr;
    ListNode* curr = head;
    while (curr) {
        ListNode* next = curr->next;
        curr->next = prev;
        prev = curr;
        curr = next;
    }
    return prev;
}

// Interview: Detect cycle
bool hasCycle(ListNode* head) {
    if (!head || !head->next) return false;
    ListNode* slow = head;
    ListNode* fast = head->next;
    while (slow != fast) {
        if (!fast || !fast->next) return false;
        slow = slow->next;
        fast = fast->next->next;
    }
    return true;
}

// Interview: Find middle
ListNode* findMiddle(ListNode* head) {
    if (!head) return nullptr;
    ListNode* slow = head;
    ListNode* fast = head;
    while (fast->next && fast->next->next) {
        slow = slow->next;
        fast = fast->next->next;
    }
    return slow;
}

void printList(ListNode* head) {
    while (head) {
        std::cout << head->val;
        if (head->next) std::cout << " -> ";
        head = head->next;
    }
    std::cout << std::endl;
}

int main() {
    ListNode* head = new ListNode(1);
    head->next = new ListNode(2);
    head->next->next = new ListNode(3);
    head->next->next->next = new ListNode(4);
    
    std::cout << "Original: ";
    printList(head);
    
    head = reverseList(head);
    std::cout << "Reversed: ";
    printList(head);
    
    return 0;
}
