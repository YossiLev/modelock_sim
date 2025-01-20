function getFieldFloat(id, defaultValue) {
    const cont = document.getElementById(id);
    if (cont != null) {
        const value = cont.value;
        const numVal = parseFloat(value);
        if (!isNaN(numVal)) {
            return numVal;
        }
    }

    return defaultValue;
}

function setFieldFloat(id, newValue) {
    const cont = document.getElementById(id);
    if (cont != null) {
        cont.value = `${newValue}`;
    }
}

function getFieldInt(id, defaultValue) {
    const cont = document.getElementById(id);
    if (cont != null) {
        const value = cont.value;
        const numVal = parseInt(value);
        if (!isNaN(numVal)) {
            return numVal;
        }
    }

    return defaultValue;
}

function setFieldInt(id, newValue) {
    const cont = document.getElementById(id);
    if (cont != null) {
        cont.value = `${newValue}`;
    }
}