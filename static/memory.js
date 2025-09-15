// Stores an object in local storage under the provided key
function saveToLocalStorage(key, object) {
    try {
      const serialized = JSON.stringify(object);
      localStorage.setItem(key, serialized);
    } catch (error) {
      console.error(`Failed to save key "${key}" to local storage:`, error);
    }
  }
  
// Retrieves an object from local storage using the provided key
function loadFromLocalStorage(key) {
    try {
      const serialized = localStorage.getItem(key);
      return serialized ? JSON.parse(serialized) : null;
    } catch (error) {
      console.error(`Failed to load key "${key}" from local storage:`, error);
      return null;
    }
}

function loadSafeNamedObjects() {
  let namedObjects = loadFromLocalStorage("namedObjects");
  if (namedObjects == null || !Array.isArray(namedObjects)) {
    namedObjects = [];
  }

  return namedObjects;
}

function addNamedObject(obj) {
  let namedObjects = loadSafeNamedObjects();

  namedObjects.push({name: obj.name, date : Date.now(), state: 0, obj: obj});

  saveToLocalStorage("namedObjects", namedObjects);
}

function mergeToNamedObject(objs) {
  let namedObjects = loadSafeNamedObjects();

  namedObjects = namedObjects.concat(objs);

  saveToLocalStorage("namedObjects", namedObjects);
}


function getNamedObjectByIndex(index) {
  let namedObjects = loadSafeNamedObjects();

  if (namedObjects.length < index) {
    return null;
  }
  return namedObjects[index].obj;
}

function deleteNamedObjectByIndex(index) {
  let namedObjects = loadSafeNamedObjects();

  namedObjects.splice(index, 1);

  saveToLocalStorage("namedObjects", namedObjects);
}
  
function AbcdMatStore(a, b, c, d) {
    saveToLocalStorage("AbcdMat", {A:a, B:b, C:c, D:d});
}

function AbcdMatCopy(name) {
    let A = document.getElementById(name + "_A").value;
    let B = document.getElementById(name + "_B").value;
    let C = document.getElementById(name + "_C").value;
    let D = document.getElementById(name + "_D").value; 

    AbcdMatStore(A, B, C, D);
}

function AbcdMatPaste(name) {
    let AbcdMat = loadFromLocalStorage("AbcdMat");
    if (AbcdMat) {
        document.getElementById(name + "_A").value = AbcdMat.A;
        document.getElementById(name + "_B").value = AbcdMat.B;
        document.getElementById(name + "_C").value = AbcdMat.C;
        document.getElementById(name + "_D").value = AbcdMat.D;
        document.getElementById(name + "_A").dispatchEvent(new Event('input'));
        document.getElementById(name + "_B").dispatchEvent(new Event('input'));
        document.getElementById(name + "_C").dispatchEvent(new Event('input'));
        document.getElementById(name + "_D").dispatchEvent(new Event('input'));
        validateMatName(name);
    } else {
        console.error("No data found in local storage for key 'AbcdMat'");
    }
}
