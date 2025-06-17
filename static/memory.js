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

function addNamedObject(obj) {
  let namedObjects = loadFromLocalStorage("namedObjects");
  if (namedObjects == null) {
    namedObjects = {names:[], dates: [], objects: []};
  }
  let objName = obj.name;
  let objDate = Date.now();
  namedObjects.names.push(objName);
  namedObjects.dates.push(objDate);
  namedObjects.objects.push(obj);

  saveToLocalStorage("namedObjects", namedObjects);
}

function mergeToNamedObject(objs) {
  let namedObjects = loadFromLocalStorage("namedObjects");
  if (namedObjects == null) {
    namedObjects = {names:[], dates: [], objects: []};
  }
  namedObjects.names = namedObjects.names.concat(objs.names);
  namedObjects.dates = namedObjects.dates.concat(objs.dates);
  namedObjects.objects = namedObjects.objects.concat(objs.objects);

  saveToLocalStorage("namedObjects", namedObjects);
}

function getNamedObjectsData() {
  let namedObjects = loadFromLocalStorage("namedObjects");
  if (namedObjects == null) {
    namedObjects = {names:[], dates:[], objects: []};
  }
  return [namedObjects.names, namedObjects.dates];
}

function getNamedObjectByIndex(index) {
  let namedObjects = loadFromLocalStorage("namedObjects");
  if (namedObjects == null) {
    namedObjects = {names:[], dates:[], objects: []};
  }
  if (namedObjects.objects.length < index) {
    return null;
  }
  return namedObjects.objects[index];
}

function deleteNamedObjectByIndex(index) {
  let namedObjects = loadFromLocalStorage("namedObjects");
  if (namedObjects == null) {
    namedObjects = {names:[], dates:[], objects: []};
  }
  namedObjects.names.splice(index, 1);
  namedObjects.dates.splice(index, 1);
  namedObjects.objects.splice(index, 1);

  saveToLocalStorage("namedObjects", namedObjects);
}
  
function AbcdMatCopy(name) {
    let A = document.getElementById(name + "_A").value;
    let B = document.getElementById(name + "_B").value;
    let C = document.getElementById(name + "_C").value;
    let D = document.getElementById(name + "_D").value; 

    saveToLocalStorage("AbcdMat", {A:A, B:B, C:C, D:D});
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
