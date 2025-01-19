function moverShow(e, innerHtml) {
    let mover = document.getElementById("globalMover");
    if (mover == null) {
        mover = document.createElement('span');
        mover.id = 'globalMover';
        document.querySelector('body').appendChild(mover);
    }
    let pos = e.target.getBoundingClientRect();

    mover.style.display = "block";
    mover.style.position = "fixed";
    mover.style.zIndex = "5";
    mover.style.background = "rgba(255, 255, 120, 0.6)";
    mover.style.padding = "5px";
    mover.style.borderRadius = "4px";
    mover.style.left = (e.clientX + 10) + 'px'; 
    mover.style.top = (e.clientY + 10) + 'px';
    //mover.style.pointerEvents = "none";
    
    mover.innerHTML = innerHtml;
    console.log("mover up")
}

function moverHide() {
    let mover = document.getElementById("globalMover");
    if (mover != null) {
        
        setTimeout(() => {
            mover.style.display = "none";
            console.log("mover down")
        }, 2000);

    }
}