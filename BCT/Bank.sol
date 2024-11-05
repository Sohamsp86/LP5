//SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;
contract Bank{
    uint256 balance = 0;
    address accHolder;

    constructor(){
        accHolder = msg.sender;
    }

    function deposit() public payable {
        require(accHolder == msg.sender, "You are not authorized to access the account");
        require(msg.value > 0 , "Deposit value should be greater than zero.");
        balance += msg.value;
    }

    function showBalance() public view returns(uint256){
        return balance;
    }

    function withdrawMoney() public payable {
        require(accHolder == msg.sender, "You are not authorized to access the account");
        require(balance > 0 , "You don't have enough balance.");
        payable(msg.sender).transfer(balance);
        balance = 0;
    }
}