class ThamudicLanguage {
    constructor() {
        this.name = 'Thamudic';
    }

    getInfo() {
        return {
            name: this.name,
            family: 'Ancient North Arabian',
            period: 'c. 8th century BCE to 4th century CE'
        };
    }
}

const thamudic = new ThamudicLanguage();
console.log('Thamudic Language Module Info:', thamudic.getInfo());

module.exports = ThamudicLanguage;
