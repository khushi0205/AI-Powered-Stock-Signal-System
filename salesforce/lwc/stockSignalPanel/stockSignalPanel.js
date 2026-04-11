import { LightningElement, api, wire, track } from 'lwc';
import getSignal from '@salesforce/apex/StockSignalController.getSignal';
import analyzeStock from '@salesforce/apex/FinancialAnalysisService.analyzeStockFromLWC';
import { refreshApex } from '@salesforce/apex';

export default class StockSignalPanel extends LightningElement {

    @api recordId;
    @track isLoading = false;

    wiredResult;
    data;

    @wire(getSignal, { recordId: '$recordId' })
    wiredSignal(result) {
        this.wiredResult = result;
        if (result.data) {
            this.data = result.data;
        }
    }

    get signalClass() {
        if (!this.data) return '';
        if (this.data.AI_Signal__c === 'BUY') return 'buy';
        if (this.data.AI_Signal__c === 'SELL') return 'sell';
        return 'hold';
    }

    handleAnalyse() {
    this.isLoading = true;

    analyzeStock({
        recordId: this.recordId,
        ticker: this.data.Stock_Ticker__c
    })
    .then(() => {
        setTimeout(() => {
            refreshApex(this.wiredResult);
            this.isLoading = false;
        }, 3000);
    })
    .catch(() => {
        this.isLoading = false;
    });
}
}