const icon = document.querySelector('header .icon');
const list = document.querySelector('header ul');
icon.addEventListener('click',()=>{
   list.classList.toggle('display');
})